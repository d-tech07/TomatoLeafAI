# Copy of the notebook Cell 5.1b (v21) architecture code. Only the export tool (tools/export_litert.py)
# imports this (it needs TensorFlow); the web server itself never does.
# CELL 5.1b - v21 architectures: LAG-HViT (Lesion-Attention-Guided Hybrid CNN-ViT) + the
# 4 single-backbone baselines, all built as  Trunk(image -> lesion_map)  +  Head(lesion_map -> outputs).
#
# WHY THE HYBRID WAS NOT BEATING THE SINGLE BACKBONES (verified causes, v9/v20):
#   (1) TRAINING BUG, hybrid only: LFSTrainerV8.train_step never called
#       optimizer.scale_loss(). Under this notebook's mixed_float16 policy
#       Keras 3 wraps every optimizer in LossScaleOptimizer, which DIVIDES
#       gradients by its dynamic loss scale (32768, doubling every 2000 clean
#       steps). With Adam that shrinks updates wherever |g/scale| ~ epsilon --
#       the reproduction in tests/test_lossscale.py shows 42% smaller weight
#       updates at LR 1e-5 after only 80 steps, and the damping grows with the
#       scale. The four baselines use model.fit()'s default train_step, which
#       DOES scale the loss -- so only the proposed model was handicapped.
#   (2) The ViT branch was a from-scratch 16x16-patch ViT (24.5% accuracy
#       on its own in the v9 run). Fusing a near-random branch adds noise.
#   (3) The lesion map / Grad-CAM target was the backbone's 7x7 stride-32
#       map, bicubically upsampled: every Grad-CAM cell covers 32x32 px, so
#       heatmaps are blobs that spill off the leaf.
#   (4) Part of the classifier's evidence bypassed the lesion map, so
#       Grad-CAM on that map could never explain the whole decision.
#
# V21 DESIGN (all 5 variants share the SAME trunk code => fair comparison):
#   Trunk  : pretrained / from-scratch backbone -> FPN-lite fusion of the
#            stride-16 and stride-32 feature maps -> lesion_map (14x14xC).
#            Native 14x14 resolution: Grad-CAM cells are 16x16 px, 4x finer.
#   Baseline head : GAP+GMP(lesion_map) -> MLP -> embedding -> logits -> pred.
#   LAG-HViT head (proposed):
#     a. lesion gate s = sigmoid(conv1x1(lesion_map))  -- supervised by the
#        pseudo-lesion target, so it is an explicit lesion probability map.
#     b. hybrid ViT: every 16x16 patch becomes a token taken from the CNN
#        lesion map (a "CNN-stem" ViT, Dosovitskiy et al. 2021), + learned
#        position embedding + [CLS] token.
#     c. Lesion-Biased Multi-Head Self-Attention: attention logits get
#        + beta_h * log(s_j) for every key token j (beta_h learned per head,
#        >= 0). The CNN's lesion activation therefore CONTROLS where the
#        transformer looks -- Novel Contribution (1) of the proposal,
#        implemented literally inside attention rather than as a pooled gate.
#     d. lesion-attention pooling of the ViT tokens + [CLS] + CNN global
#        vector -> GatedFusion (learned per-image softmax weights) -> head.
#     e. deep supervision: aux_cnn_pred (CNN branch alone) and aux_vit_pred
#        (ViT branch alone) heads -- keeps both branches individually strong
#        and gives the H2 "CNN branch vs ViT branch vs fusion" comparison
#        from one trained model.
#     f. aux_leaf_head (leaf mask) and aux_pseudo_lesion_head (= the gate s).
#   ALL class evidence flows through lesion_map (no bypass), so Grad-CAM on
#   lesion_map explains the complete decision, and the Leaf-Focus-Score
#   regulariser (Cell 7.1b) can act on the true explanation.
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, regularizers
from keras import applications as kapp

PKG = 'TomatoLeafAI_v21'
V21_BACKBONES = ('CNN', 'VGG16', 'RegNetY008', 'EfficientNetB7')


def _reg(wd):
    return regularizers.l2(wd) if wd else None


@keras.saving.register_keras_serializable(package=PKG)
class BackbonePreprocessV21(layers.Layer):
    """[0,1] RGB -> the input convention each backbone's weights expect."""

    def __init__(self, kind, **kw):
        super().__init__(**kw)
        self.kind = kind

    def call(self, x):
        x255 = x * 255.0
        if self.kind == 'VGG16':
            return kapp.vgg16.preprocess_input(x255)
        if self.kind.startswith('EfficientNet'):
            return x255                     # EfficientNet rescales/normalises internally
        return x                            # from-scratch CNN / RegNet: [0,1]

    def get_config(self):
        return {**super().get_config(), 'kind': self.kind}


def _pick_stride_layers(model, img_hw):
    """Returns (name_of_last_stride16_layer, name_of_output_layer).
    Prefers residual-sum / block-output layers so we take a finished block
    output, not an expansion conv inside the next block."""
    t16 = img_hw[0] // 16
    cands = []
    for l in model.layers:
        try:
            shp = l.output.shape
        except Exception:
            continue
        if len(shp) == 4 and shp[1] == t16:
            cands.append(l.name)
    if not cands:
        raise ValueError(f'No stride-16 layer found in {model.name}')
    preferred = [n for n in cands if n.endswith(('_add', 'conv3', '_out', '_relu3', 'project_bn'))]
    return (preferred or cands)[-1]


def _backbone_features(kind, img_size, weights, weight_decay):
    """Model(x_preprocessed) -> [f16 (H/16), f32 (H/32) or None]"""
    shape = (*img_size, 3)
    if kind == 'VGG16':
        base = kapp.VGG16(include_top=False, weights=weights, input_shape=shape)
    elif kind == 'EfficientNetB7':
        base = kapp.EfficientNetB7(include_top=False, weights=weights, input_shape=shape)
    elif kind == 'RegNetY008':
        rin = layers.Input(shape, name='regnety008_in')
        rout = build_regnety008_backbone(rin, weight_decay=weight_decay)   # Cell 5.1
        base = Model(rin, rout, name='regnety008')
    elif kind == 'CNN':
        cin = layers.Input(shape, name='cnn_in')
        x = cin
        for bi, (f, n) in enumerate(((32, 2), (64, 2), (128, 3), (256, 3), (384, 2))):
            for i in range(n):
                x = layers.Conv2D(f, 3, padding='same', kernel_regularizer=_reg(weight_decay),
                                  name=f'cnn_b{bi + 1}_conv{i + 1}')(x)
                x = layers.BatchNormalization(name=f'cnn_b{bi + 1}_bn{i + 1}')(x)
                x = layers.Activation('relu', name=f'cnn_b{bi + 1}_relu{i + 1}')(x)
            x = layers.MaxPooling2D(2, name=f'cnn_b{bi + 1}_pool')(x)
            x = layers.SpatialDropout2D(0.1, name=f'cnn_b{bi + 1}_drop')(x) if bi >= 2 else x
        base = Model(cin, x, name='cnn')
    else:
        raise ValueError(kind)
    n16 = _pick_stride_layers(base, img_size)
    fx = Model(base.input, [base.get_layer(n16).output, base.output], name=f'{kind.lower()}_features')
    fx._pretrained = kind in ('VGG16', 'EfficientNetB7') and weights is not None
    return fx


def build_trunk_v21(kind, img_size=(224, 224), lm_ch=384, weight_decay=1e-4, weights='imagenet'):
    inp = layers.Input((*img_size, 3), name='image')
    x = BackbonePreprocessV21(kind, name=f'{kind.lower()}_preproc')(inp)
    fx = _backbone_features(kind, img_size, weights if kind in ('VGG16', 'EfficientNetB7') else None, weight_decay)
    if getattr(fx, '_pretrained', False):
        fx.trainable = False          # Stage 1: head only; Stage 2 unfreezes the top layers
    f16, f32 = fx(x)
    p = layers.Conv2D(lm_ch, 1, kernel_regularizer=_reg(weight_decay), name='fpn_lat16')(f16)
    if f32.shape[1] != f16.shape[1]:
        q = layers.Conv2D(lm_ch, 1, kernel_regularizer=_reg(weight_decay), name='fpn_lat32')(f32)
        q = layers.UpSampling2D(2, interpolation='bilinear', name='fpn_up32')(q)
        p = layers.Add(name='fpn_sum')([p, q])
    p = layers.Conv2D(lm_ch, 3, padding='same', kernel_regularizer=_reg(weight_decay), name='fpn_smooth')(p)
    p = layers.BatchNormalization(name='fpn_bn')(p)
    lm = layers.Activation('relu', name='lesion_map')(p)
    return Model(inp, lm, name=f'{kind}_Trunk')


# --------------------------------------------------------------------- layers
@keras.saving.register_keras_serializable(package=PKG)
class AddClsAndPosition(layers.Layer):
    def __init__(self, num_tokens, dim, **kw):
        super().__init__(**kw)
        self.num_tokens, self.dim = int(num_tokens), int(dim)

    def build(self, input_shape):
        self.cls = self.add_weight(name='cls', shape=(1, 1, self.dim), initializer='zeros')
        self.pos = self.add_weight(name='pos', shape=(1, self.num_tokens + 1, self.dim),
                                   initializer=keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, t):
        b = tf.shape(t)[0]
        cls = tf.cast(tf.tile(self.cls, [b, 1, 1]), t.dtype)
        return tf.concat([cls, t], axis=1) + tf.cast(self.pos, t.dtype)

    def get_config(self):
        return {**super().get_config(), 'num_tokens': self.num_tokens, 'dim': self.dim}


@keras.saving.register_keras_serializable(package=PKG)
class LesionBiasedMHSA(layers.Layer):
    """Multi-head self-attention whose logits receive an additive lesion prior:
         A_h = softmax( Q_h K_h^T / sqrt(d) + beta_h * log(s + eps) )
    s = CNN lesion probability per KEY token ([CLS] gets log(1)=0).
    beta_h = softplus(raw_h) >= 0, initialised ~= 1: each head learns how much
    to trust the CNN's lesion evidence (beta -> 0 recovers plain MHSA)."""

    def __init__(self, dim, heads=8, attn_drop=0.0, proj_drop=0.1, weight_decay=1e-4,
                 use_lesion_bias=True, **kw):
        super().__init__(**kw)
        self.dim, self.heads = int(dim), int(heads)
        self.hd = self.dim // self.heads
        self.attn_drop, self.proj_drop, self.weight_decay = attn_drop, proj_drop, weight_decay
        self.use_lesion_bias = bool(use_lesion_bias)

    def build(self, input_shape):
        r = _reg(self.weight_decay)
        self.qkv = layers.Dense(3 * self.dim, kernel_regularizer=r, name='qkv')
        self.proj = layers.Dense(self.dim, kernel_regularizer=r, name='proj')
        self.qkv.build(input_shape[0]); self.proj.build(input_shape[0])
        self.beta_raw = self.add_weight(name='beta_raw', shape=(self.heads,),
                                        initializer=keras.initializers.Constant(0.5413))  # softplus -> 1.0
        self.drop_a = layers.Dropout(self.attn_drop)
        self.drop_p = layers.Dropout(self.proj_drop)

    def call(self, inputs, training=None, return_attention=False):
        x, log_s = inputs                                   # x (B,N,D); log_s (B,N) log lesion prob
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        qkv = tf.reshape(self.qkv(x), [B, N, 3, self.heads, self.hd])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])           # 3,B,H,N,hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = tf.matmul(tf.cast(q, tf.float32), tf.cast(k, tf.float32), transpose_b=True) / np.sqrt(self.hd)
        if self.use_lesion_bias:
            beta = tf.nn.softplus(tf.cast(self.beta_raw, tf.float32))[None, :, None, None]   # explicit fp32 (mixed precision)
            logits = logits + beta * tf.cast(log_s, tf.float32)[:, None, None, :]
        attn = tf.nn.softmax(logits, axis=-1)
        attn = self.drop_a(attn, training=training)
        out = tf.matmul(tf.cast(attn, v.dtype), v)          # B,H,N,hd
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [B, N, self.dim])
        out = self.drop_p(self.proj(out), training=training)
        return (out, attn) if return_attention else out

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads, 'attn_drop': self.attn_drop,
                'proj_drop': self.proj_drop, 'weight_decay': self.weight_decay,
                'use_lesion_bias': self.use_lesion_bias}


@keras.saving.register_keras_serializable(package=PKG)
class DropPath(layers.Layer):
    def __init__(self, rate=0.0, **kw):
        super().__init__(**kw); self.rate = float(rate)

    def call(self, x, training=None):
        if not training or self.rate <= 0:
            return x
        keep = 1.0 - self.rate
        shape = tf.concat([tf.shape(x)[:1], tf.ones([tf.rank(x) - 1], tf.int32)], 0)
        m = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
        return x / keep * m

    def get_config(self):
        return {**super().get_config(), 'rate': self.rate}


@keras.saving.register_keras_serializable(package=PKG)
class LogLesionPrior(layers.Layer):
    """(B,N) lesion probs -> (B,N+1) log-prior; [CLS] gets log(1)=0."""

    def call(self, s):
        s = tf.cast(s, tf.float32)
        return tf.concat([tf.zeros_like(s[:, :1]), tf.math.log(s + 1e-4)], 1)


@keras.saving.register_keras_serializable(package=PKG)
class SplitClsTokens(layers.Layer):
    """(B,N+1,D) -> [(B,D) cls, (B,N,D) patch tokens]"""

    def call(self, z):
        return [z[:, 0], z[:, 1:]]


@keras.saving.register_keras_serializable(package=PKG)
class LesionTokenPool(layers.Layer):
    """z_les = sum_i softmax_i(tau * logit(s_i)) * token_i  (tau learned > 0)"""

    def build(self, input_shape):
        self.tau_raw = self.add_weight(name='tau_raw', shape=(), initializer=keras.initializers.Constant(0.5413))

    def call(self, inputs):
        tok, s = inputs                                      # (B,N,D), (B,N)
        s = tf.clip_by_value(tf.cast(s, tf.float32), 1e-4, 1 - 1e-4)
        w = tf.nn.softmax(tf.nn.softplus(tf.cast(self.tau_raw, tf.float32)) * tf.math.log(s / (1 - s)), axis=-1)
        return tf.reduce_sum(tf.cast(w, tok.dtype)[..., None] * tok, axis=1)


@keras.saving.register_keras_serializable(package=PKG)
class GatedFusionV21(layers.Layer):
    """Per-image softmax weights over N same-width vectors."""

    def __init__(self, weight_decay=1e-4, **kw):
        super().__init__(**kw); self.weight_decay = weight_decay

    def build(self, input_shape):
        self.scorer = layers.Dense(len(input_shape), kernel_regularizer=_reg(self.weight_decay), name='scorer')
        self.scorer.build((input_shape[0][0], sum(s[-1] for s in input_shape)))

    def call(self, inputs):
        w = tf.nn.softmax(tf.cast(self.scorer(tf.concat(inputs, -1)), tf.float32), -1)
        st = tf.stack(inputs, 1)
        return tf.reduce_sum(tf.cast(w, st.dtype)[..., None] * st, 1)

    def get_config(self):
        return {**super().get_config(), 'weight_decay': self.weight_decay}


def _mlp_head(x, num_classes, dropout, wd, prefix=''):
    x = layers.Dense(512, kernel_regularizer=_reg(wd), name=f'{prefix}fc512')(x)
    x = layers.BatchNormalization(name=f'{prefix}fc_bn')(x)
    x = layers.Activation('relu', name=f'{prefix}fc_relu')(x)
    x = layers.Dropout(dropout, name=f'{prefix}fc_drop')(x)
    emb = layers.Dense(256, activation='relu', kernel_regularizer=_reg(wd), name=f'{prefix}embedding')(x)
    x = layers.Dropout(dropout * 0.6, name=f'{prefix}emb_drop')(emb)
    logits = layers.Dense(num_classes, dtype='float32', kernel_regularizer=_reg(wd), name=f'{prefix}logits')(x)
    pred = layers.Activation('softmax', dtype='float32', name=f'{prefix}pred')(logits)
    return emb, logits, pred


def build_baseline_head_v21(lm_shape, num_classes, dropout=0.5, weight_decay=1e-4):
    lm = layers.Input(lm_shape, name='lesion_map_in')
    g = layers.Concatenate(name='gap_gmp')([layers.GlobalAveragePooling2D(name='gap')(lm),
                                            layers.GlobalMaxPooling2D(name='gmp')(lm)])
    emb, logits, pred = _mlp_head(g, num_classes, dropout, weight_decay)
    return Model(lm, {'pred': pred, 'logits': logits, 'embedding': emb}, name='BaselineHead')


def build_lag_hvit_head(lm_shape, num_classes, dim=256, depth=4, heads=8, mlp_ratio=4,
                        dropout=0.5, drop_path=0.1, weight_decay=1e-4,
                        use_lesion_bias=True, use_lesion_pool=True, use_gated_fusion=True,
                        vit_only=False):
    """The proposed LAG-HViT head. Flags exist ONLY to build the H2 ablations
    from identical code: naive_concat = (use_lesion_bias=False,
    use_lesion_pool=False, use_gated_fusion=False); vit_branch = vit_only=True."""
    lm = layers.Input(lm_shape, name='lesion_map_in')
    hh, ww, ch = lm_shape
    n = hh * ww
    wd = weight_decay
    s_map = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32', name='aux_pseudo_lesion_head')(lm)
    leaf_map = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32', name='aux_leaf_head')(lm)
    s_tok = layers.Reshape((n,), name='lesion_prob_tokens')(s_map)
    log_s = LogLesionPrior(name='log_lesion_prior')(s_tok)
    t = layers.Reshape((n, ch), name='patch_tokens')(lm)
    t = layers.Dense(dim, kernel_regularizer=_reg(wd), name='token_embed')(t)
    t = AddClsAndPosition(n, dim, name='cls_pos')(t)
    for i in range(depth):
        a = layers.LayerNormalization(epsilon=1e-6, name=f'vit{i}_ln1')(t)
        a = LesionBiasedMHSA(dim, heads, weight_decay=wd, use_lesion_bias=use_lesion_bias,
                             name=f'vit{i}_lbmhsa')([a, log_s])
        t = layers.Add(name=f'vit{i}_res1')([t, DropPath(drop_path * (i + 1) / depth, name=f'vit{i}_dp1')(a)])
        m = layers.LayerNormalization(epsilon=1e-6, name=f'vit{i}_ln2')(t)
        m = layers.Dense(dim * mlp_ratio, activation='gelu', kernel_regularizer=_reg(wd), name=f'vit{i}_fc1')(m)
        m = layers.Dropout(0.1, name=f'vit{i}_mdrop')(m)
        m = layers.Dense(dim, kernel_regularizer=_reg(wd), name=f'vit{i}_fc2')(m)
        t = layers.Add(name=f'vit{i}_res2')([t, DropPath(drop_path * (i + 1) / depth, name=f'vit{i}_dp2')(m)])
    t = layers.LayerNormalization(epsilon=1e-6, name='vit_ln_out')(t)
    z_cls, z_tok = SplitClsTokens(name='vit_split_cls')(t)
    g = layers.Concatenate(name='gap_gmp')([layers.GlobalAveragePooling2D(name='gap')(lm),
                                            layers.GlobalMaxPooling2D(name='gmp')(lm)])
    g = layers.Dense(dim, activation='relu', kernel_regularizer=_reg(wd), name='cnn_global')(g)
    outs = {}
    if vit_only:
        fused = z_cls
    else:
        if use_lesion_pool:
            z_les = LesionTokenPool(name='lesion_token_pool')([z_tok, s_tok])
        else:
            z_les = layers.GlobalAveragePooling1D(name='vit_token_gap')(z_tok)
        if use_gated_fusion:
            fused = GatedFusionV21(wd, name='gated_fusion')([g, z_cls, z_les])
            fused = layers.Concatenate(name='fusion')([g, fused])
        else:
            fused = layers.Concatenate(name='fusion')([g, z_cls, z_les])
        outs['aux_cnn_pred'] = layers.Dense(num_classes, activation='softmax', dtype='float32',
                                            name='aux_cnn_pred')(layers.Dropout(0.3)(g))
        outs['aux_vit_pred'] = layers.Dense(num_classes, activation='softmax', dtype='float32',
                                            name='aux_vit_pred')(layers.Dropout(0.3)(z_cls))
    emb, logits, pred = _mlp_head(fused, num_classes, dropout, wd)
    outs.update({'pred': pred, 'logits': logits, 'embedding': emb,
                 'aux_leaf_head': leaf_map, 'aux_pseudo_lesion_head': s_map})
    return Model(lm, outs, name='LAGHViT_Head' if not vit_only else 'ViTBranch_Head')


V21_VARIANTS = {
    # variant key          : (backbone kind or None=winner, head type)
    'cnn_only':             ('CNN', 'baseline'),
    'vgg16':                ('VGG16', 'baseline'),
    'regnety008':           ('RegNetY008', 'baseline'),
    'efficientnetb7':       ('EfficientNetB7', 'baseline'),
    'proposed_hybrid':      (None, 'lag_hvit'),
    # optional H2 ablations (CFG['RUN_H2_ABLATIONS'])
    'naive_concat':         (None, 'naive_concat'),
    'vit_branch':           (None, 'vit_branch'),
}


def build_model_v21(variant, num_classes, backbone_kind=None, img_size=(224, 224), lm_ch=384,
                    dropout=0.5, weight_decay=1e-4, weights='imagenet', head_kw=None):
    """Returns (trunk, head, inference_model). inference_model: image -> pred
    (single output, what gets saved / served / evaluated)."""
    kind, head_type = V21_VARIANTS[variant]
    kind = kind or backbone_kind
    if kind is None:
        raise ValueError(f'{variant} needs backbone_kind')
    trunk = build_trunk_v21(kind, img_size, lm_ch, weight_decay, weights)
    lm_shape = tuple(int(d) for d in trunk.output.shape[1:])
    hk = dict(head_kw or {})
    if head_type == 'baseline':
        head = build_baseline_head_v21(lm_shape, num_classes, dropout, weight_decay)
    elif head_type == 'lag_hvit':
        head = build_lag_hvit_head(lm_shape, num_classes, dropout=dropout, weight_decay=weight_decay, **hk)
    elif head_type == 'naive_concat':
        head = build_lag_hvit_head(lm_shape, num_classes, dropout=dropout, weight_decay=weight_decay,
                                   use_lesion_bias=False, use_lesion_pool=False, use_gated_fusion=False, **hk)
    elif head_type == 'vit_branch':
        head = build_lag_hvit_head(lm_shape, num_classes, dropout=dropout, weight_decay=weight_decay,
                                   vit_only=True, use_lesion_bias=False, **hk)
    else:
        raise ValueError(head_type)
    img = layers.Input((*img_size, 3), name='image_input')
    out = head(trunk(img))
    inference = Model(img, out['pred'], name=variant)
    return trunk, head, inference


def find_trunk_and_head(model):
    """For a saved/loaded v21 inference model: returns (trunk, head)."""
    trunk = next(l for l in model.layers if l.name.endswith('_Trunk'))
    head = next(l for l in model.layers if l.name.endswith('_Head') or l.name == 'BaselineHead')
    return trunk, head


def get_backbone_feature_model(trunk):
    return next(l for l in trunk.layers if l.name.endswith('_features'))


def unfreeze_top_v21(trunk, top_n):
    """Stage 2: unfreeze the top `top_n` layers of a PRETRAINED backbone
    (BatchNorm stays frozen). From-scratch backbones are always trainable."""
    fx = get_backbone_feature_model(trunk)
    if not getattr(fx, '_pretrained', False) and not any(k in fx.name for k in ('vgg16', 'efficientnet')):
        return 0
    fx.trainable = True
    ls = fx.layers
    cut = max(0, len(ls) - int(top_n)) if top_n > 0 else len(ls)
    n = 0
    for i, l in enumerate(ls):
        l.trainable = (i >= cut) and not isinstance(l, layers.BatchNormalization)
        n += int(l.trainable and bool(l.weights))
    return n
