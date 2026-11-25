//
// Tooth potential for RaspberryInteraction's hybridizing-lock mode (PATCHY_lock_model = hybrid).
//
// A locked pair of patches (one sticky-end pair, "tooth") has energy T * (h * V + f(h)), where h in [0, 1] is the
// lock's hybridization level and f the sequence's hybridization free energy (both in kT). This file is V: the tooth's
// strain free energy as a function of the patch-to-patch vector r and the two patches' surface normals (a1). It
// mirrors DirectionalTether in pypatchy assays/raspberry/hybrid_model.py exactly (same formulas, same ln Z table):
//
//   d = |r|,  g_i = r.n_i,  g_j = -r.n_j          (gaps: how far each anchor sits above the other's surface)
//   c = g / sqrt(d^2 + eps_c^2)                   (softened tilt cosine; no 1/d force at contact)
//   a(g, c) = k_g/2 max(g0 - g, 0)^2 - alpha c    (squeezing wall + tilt preference)
//   W(d) = k_in/2 min(d - d_in, 0)^2 + k_out/2 x^2 + k4/4 x^4,  x = max(d - d_out, 0)
//   u = W(d) + ln Z(d) + (a(g_i, c_i) + a(g_j, c_j)) / 2,   Z(d) = int_0^1 exp(-a(d c, s(d) c)) dc
//   V = smoothcap(u): C1, flat at v_max for u >= v_max + w. A lock exists only while u < v_max + w.
//
// The parameters come from single-tooth oxDNA free energies (S6a handle pair, anchors on repulsive surfaces) and are
// given in simulation units; pypatchy's hybrid_input_keys() converts them.
//

#ifndef RASPBERRYHYBRIDTOOTH_H
#define RASPBERRYHYBRIDTOOTH_H

#include <cmath>
#include <vector>
#include "defs.h"

struct RaspberryHybridTooth {
    // tooth potential parameters (simulation units; energies in kT)
    number d_in = 0, d_out = 0, k_in = 0, k_out = 0, k4 = 0;
    number k_g = 0, g0 = 0, alpha = 0, eps_c = 0;
    number v_max = 60, cap_w = 2;
    // ln Z table
    number lnz_dmax = 0;
    int lnz_nodes = 4097, lnz_quad = 64;
    std::vector<number> lnz_val, lnz_der;
    number lnz_step = 0;
    // derived
    number d_cut = 0;     // largest span at which a lock can exist (tooth along both normals)
    number v_min = 0;

    number s_of(number d) const { return eps_c > 0 ? d / std::sqrt(d * d + eps_c * eps_c) : 1.; }
    number ds_of(number d) const { return eps_c > 0 ? eps_c * eps_c / std::pow(d * d + eps_c * eps_c, 1.5) : 0.; }

    number W(number d, number &dW) const {
        number xi = d < d_in ? d - d_in : 0.;
        number xo = d > d_out ? d - d_out : 0.;
        dW = k_in * xi + k_out * xo + k4 * xo * xo * xo;
        return 0.5 * k_in * xi * xi + 0.5 * k_out * xo * xo + 0.25 * k4 * xo * xo * xo * xo;
    }

    // Gauss-Legendre nodes and weights on [0, 1] (Newton iteration on P_n; matches numpy.polynomial.legendre.leggauss)
    static void gauss_legendre_01(int n, std::vector<number> &x, std::vector<number> &w) {
        x.assign(n, 0.);
        w.assign(n, 0.);
        for(int i = 0; i < n; i++) {
            double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 0;
            for(int it = 0; it < 100; it++) {
                double p1 = 1., p2 = 0.;
                for(int j = 1; j <= n; j++) {
                    double p3 = p2;
                    p2 = p1;
                    p1 = ((2. * j - 1.) * z * p2 - (j - 1.) * p3) / j;
                }
                pp = n * (z * p1 - p2) / (z * z - 1.);
                double z1 = z;
                z = z1 - p1 / pp;
                if(std::fabs(z - z1) < 1e-15) break;
            }
            // node on [-1, 1] is z; map to [0, 1]
            x[i] = 0.5 * (1. - z);
            w[i] = 1. / ((1. - z * z) * pp * pp);   // = 0.5 * 2 / ((1 - z^2) P'^2)
        }
    }

    // ln Z(d) and d ln Z / dd by quadrature: [0, c*] (wall active) and [c*, 1], c* = min(1, g0 / d)
    void lnz_quadrature(number d, number &lnz, number &dlnz, const std::vector<number> &x,
                        const std::vector<number> &w) const {
        number cstar = d > g0 ? g0 / d : 1.;
        number s = s_of(d), ds = ds_of(d);
        double Z = 0., dZ = 0.;
        number lims[3] = {0., cstar, 1.};
        for(int piece = 0; piece < 2; piece++) {
            number lo = lims[piece], hi = lims[piece + 1];
            for(size_t k = 0; k < x.size(); k++) {
                number c = lo + (hi - lo) * x[k];
                number wall = g0 - d * c > 0 ? g0 - d * c : 0.;
                double e = std::exp(-(0.5 * k_g * wall * wall - alpha * s * c));
                double jw = (hi - lo) * w[k];
                Z += e * jw;
                dZ += e * jw * (k_g * wall * c + alpha * ds * c);
            }
        }
        lnz = std::log(Z);
        dlnz = dZ / Z;
    }

    void build() {
        std::vector<number> x, w;
        gauss_legendre_01(lnz_quad, x, w);
        lnz_val.resize(lnz_nodes);
        lnz_der.resize(lnz_nodes);
        lnz_step = lnz_dmax / (lnz_nodes - 1);
        for(int k = 0; k < lnz_nodes; k++) {
            lnz_quadrature(k * lnz_step, lnz_val[k], lnz_der[k], x, w);
        }
        // cutoff and minimum, scanning the table nodes with the tooth along both normals (g_i = g_j = d)
        d_cut = 0;
        v_min = 1e30;
        int last_below = -1;
        for(int k = 0; k < lnz_nodes; k++) {
            number d = k * lnz_step;
            number u = best_orientation_u(d);
            if(u < v_max + cap_w) last_below = k;
            if(u < v_min) v_min = u;
        }
        int kc = last_below + 1 < lnz_nodes ? last_below + 1 : lnz_nodes - 1;
        d_cut = kc * lnz_step;
        v_min = std::floor(v_min) - 1.;
    }

    // cubic Hermite interpolation of the table
    number lnz(number d, number &dlnz) const {
        number xx = d / lnz_step;
        if(xx < 0) xx = 0;
        if(xx > lnz_nodes - 1 - 1e-12) xx = lnz_nodes - 1 - 1e-12;
        int k = (int) xx;
        number t = xx - k, t2 = t * t, t3 = t2 * t;
        number y0 = lnz_val[k], y1 = lnz_val[k + 1];
        number m0 = lnz_der[k] * lnz_step, m1 = lnz_der[k + 1] * lnz_step;
        dlnz = ((6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * m0 + (-6 * t2 + 6 * t) * y1 + (3 * t2 - 2 * t) * m1) / lnz_step;
        return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1;
    }

    number a(number g, number c) const {
        number wall = g0 - g > 0 ? g0 - g : 0.;
        return 0.5 * k_g * wall * wall - alpha * c;
    }

    number best_orientation_u(number d) const {
        number dW, dl;
        number rs = std::sqrt(d * d + eps_c * eps_c);
        if(rs < 1e-12) rs = 1e-12;
        return W(d, dW) + lnz(d, dl) + a(d, d / rs);
    }

    // C1 soft cap and its derivative
    number smoothcap(number u, number &dcap) const {
        if(cap_w <= 0) {
            dcap = u < v_max ? 1. : 0.;
            return u < v_max ? u : v_max;
        }
        if(u <= v_max - cap_w) {
            dcap = 1.;
            return u;
        }
        if(u >= v_max + cap_w) {
            dcap = 0.;
            return v_max;
        }
        number y = u - v_max + cap_w;
        dcap = 1. - y / (2 * cap_w);
        return u - y * y / (4 * cap_w);
    }

    /**
     * tooth energy V (kT) for patch-to-patch vector r (from patch i to patch j) and surface normals ni, nj; u is the
     * uncapped value (lock exists iff u < v_max + cap_w). With want_grad, fills dV/dr, dV/dni, dV/dnj.
     */
    number energy(const LR_vector &r, const LR_vector &ni, const LR_vector &nj, number &u, bool want_grad,
                  LR_vector &dV_dr, LR_vector &dV_dni, LR_vector &dV_dnj) const {
        number d = r.module();
        number gi = r * ni, gj = -(r * nj);
        number rs = std::sqrt(d * d + eps_c * eps_c);
        if(rs < 1e-12) rs = 1e-12;
        number ci = gi / rs, cj = gj / rs;
        number dW, dl;
        u = W(d, dW) + lnz(d, dl) + 0.5 * (a(gi, ci) + a(gj, cj));
        number dcap;
        number V = smoothcap(u, dcap);
        if(want_grad) {
            number wi = g0 - gi > 0 ? g0 - gi : 0., wj = g0 - gj > 0 ? g0 - gj : 0.;
            number dagi = -k_g * wi, dagj = -k_g * wj;      // da/dg
            number dac = -alpha;                             // da/dc
            LR_vector rhat = d > 1e-12 ? r / d : LR_vector(0., 0., 0.);
            number rs3 = rs * rs * rs;
            // u = W + lnZ + (a_i + a_j)/2; dg_i/dr = ni, dg_j/dr = -nj; dc_i/dr = ni/rs - g_i r/rs^3, dc_j/dr = -nj/rs - g_j r/rs^3
            LR_vector du_dr = rhat * (dW + dl)
                    + (ni * dagi + (ni / rs - r * (gi / rs3)) * dac) * 0.5
                    + (nj * (-dagj) + (nj * (-1. / rs) - r * (gj / rs3)) * dac) * 0.5;
            LR_vector du_dni = (r * dagi + r * (dac / rs)) * 0.5;
            LR_vector du_dnj = (r * (-dagj) + r * (-dac / rs)) * 0.5;
            dV_dr = du_dr * dcap;
            dV_dni = du_dni * dcap;
            dV_dnj = du_dnj * dcap;
        }
        return V;
    }
};

#endif // RASPBERRYHYBRIDTOOTH_H
