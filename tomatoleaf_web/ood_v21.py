# Web copy of notebook Cell 9.5 (module part). The web app only uses from_state() + pvalues().
# CELL 9.5 (module) - Feature-space OOD for the tomato-leaf gate: class-conditional
# Mahalanobis + Relative Mahalanobis (Ren et al., 2021) on the classifier's embedding,
# and the energy score (Liu et al., 2020) on its logits. Scores are turned into
# calibrated "in-distribution p-values" with the empirical CDF of TOMATO VALIDATION
# images, so a threshold of 0.05 means "accept 95% of real tomato leaves".
import numpy as np


class FeatureOODV21:
    def fit(self, emb, y):
        from sklearn.covariance import LedoitWolf      # training-time only
        emb = np.asarray(emb, np.float64); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([emb[y == c].mean(0) for c in self.classes_])
        centered = np.concatenate([emb[y == c] - self.means_[i] for i, c in enumerate(self.classes_)])
        self.prec_ = LedoitWolf().fit(centered).precision_
        self.mu0_ = emb.mean(0)
        self.prec0_ = LedoitWolf().fit(emb - self.mu0_).precision_
        return self

    def distances(self, emb):
        emb = np.asarray(emb, np.float64)
        d = np.stack([np.einsum('nd,dk,nk->n', emb - m, self.prec_, emb - m) for m in self.means_], 1)
        d0 = np.einsum('nd,dk,nk->n', emb - self.mu0_, self.prec0_, emb - self.mu0_)
        return {'maha': d.min(1), 'rmd': (d - d0[:, None]).min(1)}

    @staticmethod
    def energy(logits, T=1.0):
        z = np.asarray(logits, np.float64) / T
        m = z.max(1, keepdims=True)
        return -(T * (m[:, 0] + np.log(np.exp(z - m).sum(1))))      # lower = more in-distribution

    def calibrate(self, emb_val, logits_val):
        d = self.distances(emb_val)
        self.ref_ = {'rmd': np.sort(d['rmd']), 'energy': np.sort(self.energy(logits_val))}
        return self

    def pvalues(self, emb, logits):
        """p = fraction of tomato-validation images that look LESS typical
        than this one. ~Uniform(0,1) for real tomato leaves; ~0 for OOD."""
        d = self.distances(emb)
        out = {}
        for k, v in (('maha_score', d['rmd']), ('energy_score', self.energy(logits))):
            ref = self.ref_['rmd' if k == 'maha_score' else 'energy']
            out[k] = 1.0 - np.searchsorted(ref, v, side='left') / len(ref)
        return out

    def state(self):
        return {'classes': self.classes_.tolist(), 'means': self.means_.tolist(), 'prec': self.prec_.tolist(),
                'mu0': self.mu0_.tolist(), 'prec0': self.prec0_.tolist(),
                'ref_rmd': self.ref_['rmd'].tolist(), 'ref_energy': self.ref_['energy'].tolist()}

    @classmethod
    def from_state(cls, s):
        o = cls()
        o.classes_ = np.array(s['classes']); o.means_ = np.array(s['means']); o.prec_ = np.array(s['prec'])
        o.mu0_ = np.array(s['mu0']); o.prec0_ = np.array(s['prec0'])
        o.ref_ = {'rmd': np.array(s['ref_rmd']), 'energy': np.array(s['ref_energy'])}
        return o


def ood_report(scores_in, scores_out):
    """scores: higher = more tomato-like. Returns AUROC and FPR@95%TPR."""
    from sklearn.metrics import roc_auc_score
    s = np.concatenate([scores_in, scores_out]); l = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    thr = np.quantile(scores_in, 0.05)
    return {'auroc': float(roc_auc_score(l, s)), 'fpr_at_95tpr': float((np.asarray(scores_out) >= thr).mean()),
            'n_in': int(len(scores_in)), 'n_out': int(len(scores_out))}
