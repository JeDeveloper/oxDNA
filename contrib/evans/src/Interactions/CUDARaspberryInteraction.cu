//
// CUDARaspberryInteraction.cu
// Created by josh evans on 3 December 2024.
//
// CUDA implementation of RaspberryInteraction.
// Modelled after CUDADetailedPatchySwapInteraction (contrib/rovigatti).
//
// Force/torque sign conventions (match CPU RaspberryInteraction):
//   F     : accumulates world-frame force on particle p; at end stored directly.
//   torque: accumulates world-frame torque on particle p; converted to body frame at end via
//           torques[IND] = _vectors_transpose_c_number4_product(a1, a2, a3, T).
//   For a force `f` (pointing from p toward q) and patch position pw (world frame):
//     F       -= f          (p pushed away from q)
//     torque  -= _cross(pw, f)
//
// Note: patch locking is NOT implemented on GPU (same as DPS; locking is inherently serial).
//

#include "CUDARaspberryInteraction.h"

#include "CUDA/Lists/CUDASimpleVerletList.h"
#include "CUDA/Lists/CUDANoList.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cmath>

/* ============================================================
 * GPU constant memory
 * ============================================================ */

__constant__ int   MD_N[1];
__constant__ int   MD_N_species[1];
__constant__ int   MD_N_patches [CUDARaspberryInteraction::MAX_SPECIES];
__constant__ int   MD_N_rep_pts [CUDARaspberryInteraction::MAX_SPECIES];
// patch type ID for each (species, local_patch_index) pair
__constant__ int   MD_patch_type_ids[CUDARaspberryInteraction::MAX_SPECIES]
                                    [CUDARaspberryInteraction::MAX_PATCHES];
__constant__ float MD_sqr_rcut[1];
__constant__ bool  MD_angmod[1];
// NarrowType parameters for angular modulation
__constant__ float MD_narrow_t0[1], MD_narrow_ts[1], MD_narrow_tc[1],
                   MD_narrow_a[1],  MD_narrow_b[1];
// Global patchy energy cut (computed from default alpha in RaspberryInteraction::init)
__constant__ float MD_patch_E_cut[1];

#include "CUDA/cuda_utils/CUDA_lr_common.cuh"

/* ============================================================
 * Device helper functions
 * ============================================================ */

/**
 * Angular modulation V_mod(t): parabolic well with quadratic smoothing.
 * Mirrors NarrowType::V_mod on the CPU.
 */
__device__ __forceinline__
float _V_mod(float t, float t0, float ts, float tc, float a, float b) {
    float val = 0.f;
    t -= t0;
    if (t < 0.f) t = -t;
    if (t < tc) {
        if (t > ts)
            val = b * (tc - t) * (tc - t);
        else
            val = 1.f - a * t * t;
    }
    return val;
}

/**
 * Angular modulation derivative term d(V_mod)/d(sin(t)).
 * Mirrors NarrowType::V_modDsin on the CPU.
 */
__device__ __forceinline__
float _V_modDsin(float t, float t0, float ts, float tc, float a, float b) {
    float val = 0.f;
    float m   = 1.f;
    float tt0 = t - t0;
    if (tt0 < 0.f) { tt0 = -tt0; m = -1.f; }
    if (tt0 < tc) {
        float sint = sinf(t);
        if (tt0 > ts) {
            val = m * 2.f * b * (tt0 - tc) / sint;
        } else {
            if (sint * sint > 1e-8f)
                val = -m * 2.f * a * tt0 / sint;
            else
                val = -m * 2.f * a;
        }
    }
    return val;
}

/**
 * Gaussian-well patch energy kernel.
 * Returns  -1.001 * exp(-(r/alpha)^10)  and sets r8b10 = r^8 / alpha^10.
 * Mirrors RaspberryInteraction::compute_energy on the CPU.
 */
__device__ __forceinline__
c_number _compute_patch_energy(c_number dist_sqr, c_number alpha_pow, c_number &r8b10) {
    r8b10 = (dist_sqr * dist_sqr * dist_sqr * dist_sqr) / alpha_pow;
    return (c_number) -1.001 * exp(-r8b10 * dist_sqr);
}

/* ============================================================
 * Device function: repulsive-sphere interaction
 *
 * Iterates all (pp, qq) pairs of repulsion points on particles p and q.
 * Uses LJ + quadratic-smoothing potential (PLEXCL_* constants from RaspberryInteraction.h).
 * Updates F (world-frame force) and torque (world-frame torque) for particle p.
 * ============================================================ */
__device__
void _repulsive_interaction(c_number4 &ppos, c_number4 &qpos,
                            c_number4 &a1, c_number4 &a2, c_number4 &a3,
                            c_number4 &b1, c_number4 &b2, c_number4 &b3,
                            c_number4 &r,   // minimum-image vector p→q
                            c_number4 &F, c_number4 &torque,
                            cudaTextureObject_t tex_rep_pts) {
    int ptype = get_particle_btype(ppos);
    int qtype = get_particle_btype(qpos);

    int p_nrep = MD_N_rep_pts[ptype];
    int q_nrep = MD_N_rep_pts[qtype];

    for (int pp = 0; pp < p_nrep; pp++) {
        float4 p_data = tex1Dfetch<float4>(tex_rep_pts,
                            pp + ptype * CUDARaspberryInteraction::MAX_REP_PTS);
        // Rotate body-frame position to world frame using p's orientation matrix (a1,a2,a3)
        c_number4 p_rep_pos = {
            a1.x*p_data.x + a2.x*p_data.y + a3.x*p_data.z,
            a1.y*p_data.x + a2.y*p_data.y + a3.y*p_data.z,
            a1.z*p_data.x + a2.z*p_data.y + a3.z*p_data.z, 0.f
        };
        c_number p_radius = p_data.w;

        for (int qq = 0; qq < q_nrep; qq++) {
            float4 q_data = tex1Dfetch<float4>(tex_rep_pts,
                                qq + qtype * CUDARaspberryInteraction::MAX_REP_PTS);
            c_number4 q_rep_pos = {
                b1.x*q_data.x + b2.x*q_data.y + b3.x*q_data.z,
                b1.y*q_data.x + b2.y*q_data.y + b3.y*q_data.z,
                b1.z*q_data.x + b2.z*q_data.y + b3.z*q_data.z, 0.f
            };
            c_number q_radius = q_data.w;

            c_number rsum     = p_radius + q_radius;
            c_number rmax_sqr = (PLEXCL_RC * rsum) * (PLEXCL_RC * rsum);

            // displacement vector from p_rep to q_rep (world frame)
            c_number4 rep_dist = {
                r.x + q_rep_pos.x - p_rep_pos.x,
                r.y + q_rep_pos.y - p_rep_pos.y,
                r.z + q_rep_pos.z - p_rep_pos.z, 0.f
            };
            c_number rep_dist_sqr = CUDA_DOT(rep_dist, rep_dist);

            if (rep_dist_sqr >= rmax_sqr) continue;

            c_number sigma_sqr  = (PLEXCL_S * rsum) * (PLEXCL_S * rsum);
            c_number rstar      = PLEXCL_R  * rsum;
            c_number rc         = PLEXCL_RC * rsum;
            c_number energy;
            // force_cpu = -rep_dist * coeff, where coeff > 0 means pointing p→q.
            // We store force_cpu and then do F -= force_cpu, torque -= cross(p_rep_pos, force_cpu).
            c_number force_mod; // multiply rep_dist to get force_cpu

            if (rep_dist_sqr < rstar * rstar) {
                // Full Lennard-Jones
                c_number lj = sigma_sqr / rep_dist_sqr;          // (sigma/r)^2
                lj = lj * lj * lj;                                // (sigma/r)^6
                energy     = 4.f * PLEXCL_EPS * (lj*lj - lj);
                // CPU: force = -rep_dist * 24*EPS*(lj - 2*lj^2)/r^2
                // At close range (lj >> 1): coeff < 0, so -rep_dist * negative = pointing p→q (repulsive on p via -=)
                force_mod  = -(24.f * PLEXCL_EPS * (lj - 2.f*lj*lj) / rep_dist_sqr);
            } else {
                // Quadratic smoothing (rstar ≤ r < rc)
                c_number r_rep = sqrtf(rep_dist_sqr);
                c_number rrc   = r_rep - rc;                       // < 0 in this region
                c_number b_c   = PLEXCL_B / (rsum * rsum);
                energy    = PLEXCL_EPS * b_c * rrc * rrc;
                // CPU: force = -rep_dist * 2*EPS*b*rrc/r_patch (rrc < 0 → coeff < 0)
                force_mod = -(2.f * PLEXCL_EPS * b_c * rrc / r_rep);
            }

            c_number4 force_cpu = rep_dist * force_mod;

            // Apply to p: p->force -= force_cpu  (repulsion pushes p away from q)
            F.x     -= force_cpu.x;
            F.y     -= force_cpu.y;
            F.z     -= force_cpu.z;
            F.w     += energy;

            // Torque on p in world frame: -(p_rep_pos × force_cpu)
            // CPU: p->torque += -orientationT * (p_rep_pos × force_cpu)
            // CUDA accumulates world-frame and converts at end.
            torque -= _cross(p_rep_pos, force_cpu);
        }
    }
}

/* ============================================================
 * Device function: patchy interaction WITHOUT angular modulation
 * ============================================================ */
__device__
void _patchy_noangmod_body(c_number4 &ppos, c_number4 &qpos,
                           c_number4 &a1, c_number4 &a2, c_number4 &a3,
                           c_number4 &b1, c_number4 &b2, c_number4 &b3,
                           c_number4 &r,   // min-image p→q
                           c_number4 &F, c_number4 &torque,
                           cudaTextureObject_t tex_patch_pos,
                           cudaTextureObject_t tex_pp_int) {
    int ptype   = get_particle_btype(ppos);
    int qtype   = get_particle_btype(qpos);
    int p_npatch = MD_N_patches[ptype];
    int q_npatch = MD_N_patches[qtype];

    for (int pp = 0; pp < p_npatch; pp++) {
        int p_ptid = MD_patch_type_ids[ptype][pp];

        float4 p_bp = tex1Dfetch<float4>(tex_patch_pos,
                          pp + ptype * CUDARaspberryInteraction::MAX_PATCHES);
        c_number4 p_patch_pos = {
            a1.x*p_bp.x + a2.x*p_bp.y + a3.x*p_bp.z,
            a1.y*p_bp.x + a2.y*p_bp.y + a3.y*p_bp.z,
            a1.z*p_bp.x + a2.z*p_bp.y + a3.z*p_bp.z, 0.f
        };

        for (int qq = 0; qq < q_npatch; qq++) {
            int q_ptid = MD_patch_type_ids[qtype][qq];

            float4 pp_int = tex1Dfetch<float4>(tex_pp_int,
                                p_ptid * CUDARaspberryInteraction::MAX_PATCH_TYPES + q_ptid);
            c_number eps          = pp_int.x;
            c_number alpha_pow    = pp_int.y;
            c_number max_dist_sqr = pp_int.z;

            if (eps == 0.f) continue;

            float4 q_bp = tex1Dfetch<float4>(tex_patch_pos,
                              qq + qtype * CUDARaspberryInteraction::MAX_PATCHES);
            c_number4 q_patch_pos = {
                b1.x*q_bp.x + b2.x*q_bp.y + b3.x*q_bp.z,
                b1.y*q_bp.x + b2.y*q_bp.y + b3.y*q_bp.z,
                b1.z*q_bp.x + b2.z*q_bp.y + b3.z*q_bp.z, 0.f
            };

            // Patch–patch displacement vector (world frame)
            c_number4 patch_dist = {
                r.x + q_patch_pos.x - p_patch_pos.x,
                r.y + q_patch_pos.y - p_patch_pos.y,
                r.z + q_patch_pos.z - p_patch_pos.z, 0.f
            };
            c_number dist_sqr = CUDA_DOT(patch_dist, patch_dist);

            if (dist_sqr >= max_dist_sqr) continue;

            // Gaussian well: E = eps * (-1.001) * exp(-(r/alpha)^10)
            c_number r8b10;
            c_number E0       = _compute_patch_energy(dist_sqr, alpha_pow, r8b10);
            c_number exp_part = eps * E0; // < 0 for binding

            // Force magnitude along patch_dist:  f1D = 5 * exp_part * r8b10
            // CPU: tmp_force = patch_dist * f1D;  p->force -= tmp_force
            c_number f1D      = 5.f * exp_part * r8b10;
            c_number4 tmp_force = patch_dist * f1D;

            F      -= tmp_force;
            F.w    += exp_part;
            // Torque: CPU does p->torque -= orientationT * (ppatch × tmp_force)
            // In world frame: torque -= ppatch × tmp_force
            torque -= _cross(p_patch_pos, tmp_force);
        }
    }
}

/* ============================================================
 * Device function: patchy interaction WITH angular modulation
 * ============================================================ */
__device__
void _patchy_angmod_body(c_number4 &ppos, c_number4 &qpos,
                         c_number4 &a1, c_number4 &a2, c_number4 &a3,
                         c_number4 &b1, c_number4 &b2, c_number4 &b3,
                         c_number4 &r,
                         c_number4 &F, c_number4 &torque,
                         cudaTextureObject_t tex_patch_pos,
                         cudaTextureObject_t tex_patch_ori,
                         cudaTextureObject_t tex_pp_int) {
    int ptype   = get_particle_btype(ppos);
    int qtype   = get_particle_btype(qpos);
    int p_npatch = MD_N_patches[ptype];
    int q_npatch = MD_N_patches[qtype];

    c_number sqr_r = CUDA_DOT(r, r);
    c_number rdist  = sqrtf(sqr_r);
    // Unit vector along p→q (center-to-center)
    c_number inv_r  = 1.f / rdist;
    c_number4 r_hat = { r.x*inv_r, r.y*inv_r, r.z*inv_r, 0.f };

    for (int pp = 0; pp < p_npatch; pp++) {
        int p_ptid = MD_patch_type_ids[ptype][pp];

        // Body-frame patch position → world frame
        float4 p_bp = tex1Dfetch<float4>(tex_patch_pos,
                          pp + ptype * CUDARaspberryInteraction::MAX_PATCHES);
        c_number4 p_patch_pos = {
            a1.x*p_bp.x + a2.x*p_bp.y + a3.x*p_bp.z,
            a1.y*p_bp.x + a2.y*p_bp.y + a3.y*p_bp.z,
            a1.z*p_bp.x + a2.z*p_bp.y + a3.z*p_bp.z, 0.f
        };

        // Body-frame patch orientation (a1 vector) → world frame
        float4 p_bo = tex1Dfetch<float4>(tex_patch_ori,
                          pp + ptype * CUDARaspberryInteraction::MAX_PATCHES);
        c_number4 p_patch_a1 = {
            a1.x*p_bo.x + a2.x*p_bo.y + a3.x*p_bo.z,
            a1.y*p_bo.x + a2.y*p_bo.y + a3.y*p_bo.z,
            a1.z*p_bo.x + a2.z*p_bo.y + a3.z*p_bo.z, 0.f
        };

        for (int qq = 0; qq < q_npatch; qq++) {
            int q_ptid = MD_patch_type_ids[qtype][qq];

            float4 pp_int = tex1Dfetch<float4>(tex_pp_int,
                                p_ptid * CUDARaspberryInteraction::MAX_PATCH_TYPES + q_ptid);
            c_number eps          = pp_int.x;
            c_number alpha_pow    = pp_int.y;
            c_number max_dist_sqr = pp_int.z;

            if (eps == 0.f) continue;

            float4 q_bp = tex1Dfetch<float4>(tex_patch_pos,
                              qq + qtype * CUDARaspberryInteraction::MAX_PATCHES);
            c_number4 q_patch_pos = {
                b1.x*q_bp.x + b2.x*q_bp.y + b3.x*q_bp.z,
                b1.y*q_bp.x + b2.y*q_bp.y + b3.y*q_bp.z,
                b1.z*q_bp.x + b2.z*q_bp.y + b3.z*q_bp.z, 0.f
            };

            float4 q_bo = tex1Dfetch<float4>(tex_patch_ori,
                              qq + qtype * CUDARaspberryInteraction::MAX_PATCHES);
            c_number4 q_patch_a1 = {
                b1.x*q_bo.x + b2.x*q_bo.y + b3.x*q_bo.z,
                b1.y*q_bo.x + b2.y*q_bo.y + b3.y*q_bo.z,
                b1.z*q_bo.x + b2.z*q_bo.y + b3.z*q_bo.z, 0.f
            };

            c_number4 patch_dist = {
                r.x + q_patch_pos.x - p_patch_pos.x,
                r.y + q_patch_pos.y - p_patch_pos.y,
                r.z + q_patch_pos.z - p_patch_pos.z, 0.f
            };
            c_number dist_sqr = CUDA_DOT(patch_dist, patch_dist);

            if (dist_sqr >= max_dist_sqr) continue;

            // ---------- radial (Gaussian well) part ----------
            c_number r8b10;
            c_number E0       = _compute_patch_energy(dist_sqr, alpha_pow, r8b10);
            c_number exp_part = eps * E0;

            c_number f1D         = 5.f * exp_part * r8b10;
            // tmp_force = radial force direction (used for radial torque contribution)
            c_number4 tmp_force_rad = patch_dist * f1D;

            // ---------- angular modulation ----------
            // cosa1 = p_patch_a1 · r_hat   (angle of p's patch a1 w.r.t. centre-centre vector)
            // cosb1 = -q_patch_a1 · r_hat  (note negative sign: q patch faces p)
            c_number cosa1 = CUDA_DOT(p_patch_a1, r_hat);
            c_number cosb1 = -CUDA_DOT(q_patch_a1, r_hat);

            // Clamp to avoid NaN from acosf
            cosa1 = fminf(fmaxf(cosa1, -1.f), 1.f);
            cosb1 = fminf(fmaxf(cosb1, -1.f), 1.f);

            c_number ta1 = acosf(cosa1);
            c_number tb1 = acosf(cosb1);

            c_number fa1 = _V_mod(ta1, MD_narrow_t0[0], MD_narrow_ts[0],
                                  MD_narrow_tc[0], MD_narrow_a[0], MD_narrow_b[0]);
            c_number fb1 = _V_mod(tb1, MD_narrow_t0[0], MD_narrow_ts[0],
                                  MD_narrow_tc[0], MD_narrow_a[0], MD_narrow_b[0]);

            // f1 = -eps * (exp_part - _patch_E_cut)
            // This is the angular modulation scaling (positive at binding distances)
            c_number f1      = -eps * (exp_part - MD_patch_E_cut[0]);
            c_number angmod  = f1 * fa1 * fb1;
            c_number e_ij    = exp_part * angmod;

            F.w += e_ij;

            // ---------- forces and torques ----------
            c_number fa1Dsin = _V_modDsin(ta1, MD_narrow_t0[0], MD_narrow_ts[0],
                                          MD_narrow_tc[0], MD_narrow_a[0], MD_narrow_b[0]);
            c_number fb1Dsin = _V_modDsin(tb1, MD_narrow_t0[0], MD_narrow_ts[0],
                                          MD_narrow_tc[0], MD_narrow_a[0], MD_narrow_b[0]);

            // Angular torque contributions on p (world frame):
            //   VM1: dir = r_hat × p_patch_a1; torquep += dir * (f1*fa1Dsin*fb1)
            //   then: p->torque -= orientationT * torquep
            // In CUDA (world-frame accumulation):
            //   torque -= _cross(r_hat, p_patch_a1) * (f1*fa1Dsin*fb1)
            torque -= _cross(r_hat, p_patch_a1) * (f1 * fa1Dsin * fb1);
            // Radial torque from radial force:
            //   torquep += ppatch × tmp_force_rad  →  torque -= _cross(p_patch_pos, tmp_force_rad)
            torque -= _cross(p_patch_pos, tmp_force_rad);

            // Build final force (radial + angular corrections):
            // tmp_force += (ppatch_a1 - r_hat*cosa1) * (f1*fa1Dsin*fb1/rdist)
            // tmp_force -= (qpatch_a1 + r_hat*cosb1) * (f1*fa1*fb1Dsin/rdist)
            c_number4 tmp_force = tmp_force_rad;
            c_number coeff_p = f1 * fa1Dsin * fb1  / rdist;
            c_number coeff_q = f1 * fa1     * fb1Dsin / rdist;

            tmp_force.x += (p_patch_a1.x - r_hat.x * cosa1) * coeff_p;
            tmp_force.y += (p_patch_a1.y - r_hat.y * cosa1) * coeff_p;
            tmp_force.z += (p_patch_a1.z - r_hat.z * cosa1) * coeff_p;

            tmp_force.x -= (q_patch_a1.x + r_hat.x * cosb1) * coeff_q;
            tmp_force.y -= (q_patch_a1.y + r_hat.y * cosb1) * coeff_q;
            tmp_force.z -= (q_patch_a1.z + r_hat.z * cosb1) * coeff_q;

            // CPU: p->force -= tmp_force
            F -= tmp_force;
        }
    }
}

/* ============================================================
 * Main CUDA kernel
 * ============================================================ */
__global__
void RI_forces(c_number4 *poss, GPU_quat *orientations,
               c_number4 *forces, c_number4 *torques,
               int *matrix_neighs, int *number_neighs,
               cudaTextureObject_t tex_patch_pos, cudaTextureObject_t tex_patch_ori,
               cudaTextureObject_t tex_rep_pts,   cudaTextureObject_t tex_pp_int,
               CUDABox *box) {
    if (IND >= MD_N[0]) return;

    c_number4 F      = forces  [IND];
    c_number4 T      = torques [IND];
    c_number4 ppos   = poss    [IND];
    GPU_quat  po     = orientations[IND];

    c_number4 a1, a2, a3;
    get_vectors_from_quat(po, a1, a2, a3);

    int num_neighs = NUMBER_NEIGHBOURS(IND, number_neighs);
    for (int j = 0; j < num_neighs; j++) {
        int k = NEXT_NEIGHBOUR(IND, j, matrix_neighs);
        if (k == IND) continue;

        c_number4 qpos = poss[k];
        GPU_quat  qo   = orientations[k];
        c_number4 b1, b2, b3;
        get_vectors_from_quat(qo, b1, b2, b3);

        // Minimum-image vector from p to q
        c_number4 r = box->minimum_image(ppos, qpos);
        c_number sqr_r = CUDA_DOT(r, r);
        if (sqr_r >= MD_sqr_rcut[0]) continue;

        // Repulsion
        _repulsive_interaction(ppos, qpos, a1, a2, a3, b1, b2, b3,
                               r, F, T, tex_rep_pts);

        // Patchy
        if (MD_angmod[0]) {
            _patchy_angmod_body(ppos, qpos, a1, a2, a3, b1, b2, b3,
                                r, F, T,
                                tex_patch_pos, tex_patch_ori, tex_pp_int);
        } else {
            _patchy_noangmod_body(ppos, qpos, a1, a2, a3, b1, b2, b3,
                                  r, F, T,
                                  tex_patch_pos, tex_pp_int);
        }
    }

    forces [IND] = F;
    // Convert accumulated world-frame torque to body frame
    torques[IND] = _vectors_transpose_c_number4_product(a1, a2, a3, T);
}

/* ============================================================
 * Host-side class implementation
 * ============================================================ */

CUDARaspberryInteraction::CUDARaspberryInteraction() :
        CUDABaseInteraction(), RaspberryInteraction() {}

CUDARaspberryInteraction::~CUDARaspberryInteraction() {
    if (_d_patch_pos) {
        CUDA_SAFE_CALL(cudaFree(_d_patch_pos));
        cudaDestroyTextureObject(_tex_patch_pos);
    }
    if (_d_patch_ori) {
        CUDA_SAFE_CALL(cudaFree(_d_patch_ori));
        cudaDestroyTextureObject(_tex_patch_ori);
    }
    if (_d_rep_pts) {
        CUDA_SAFE_CALL(cudaFree(_d_rep_pts));
        cudaDestroyTextureObject(_tex_rep_pts);
    }
    if (_d_pp_int) {
        CUDA_SAFE_CALL(cudaFree(_d_pp_int));
        cudaDestroyTextureObject(_tex_pp_int);
    }
}

void CUDARaspberryInteraction::get_settings(input_file &inp) {
    RaspberryInteraction::get_settings(inp);

    int sort_every = 0;
    getInputInt(&inp, "CUDA_sort_every", &sort_every, 0);
}

void CUDARaspberryInteraction::cuda_init(int N) {
    CUDABaseInteraction::cuda_init(N);

    // Populate data structures (m_ParticleTypes, m_PatchesTypes, etc.) if not already done.
    // Like CUDADetailedPatchySwapInteraction, we call read_topology with temporary particles
    // so cuda_init works regardless of whether the CPU path has run first.
    int N_strands;
    std::vector<BaseParticle *> particles(N);
    RaspberryInteraction::read_topology(&N_strands, particles);
    for (auto p : particles) delete p;

    RaspberryInteraction::init(); // sets _patch_E_cut etc.

    // ---- Sanity checks ----
    int N_species = (int)m_ParticleTypes.size();
    if (N_species > MAX_SPECIES)
        throw oxDNAException("CUDARaspberryInteraction: too many particle species (%d > %d). "
                             "Increase MAX_SPECIES.", N_species, MAX_SPECIES);

    int N_patch_types = (int)m_PatchesTypes.size();
    if (N_patch_types > MAX_PATCH_TYPES)
        throw oxDNAException("CUDARaspberryInteraction: too many patch types (%d > %d). "
                             "Increase MAX_PATCH_TYPES.", N_patch_types, MAX_PATCH_TYPES);

    for (int s = 0; s < N_species; s++) {
        int np = (int)std::get<PTYPE_PATCH_IDS>(m_ParticleTypes[s]).size();
        int nr = (int)std::get<PTYPE_REP_PTS>  (m_ParticleTypes[s]).size();
        if (np > MAX_PATCHES)
            throw oxDNAException("CUDARaspberryInteraction: species %d has %d patches > MAX_PATCHES=%d",
                                 s, np, MAX_PATCHES);
        if (nr > MAX_REP_PTS)
            throw oxDNAException("CUDARaspberryInteraction: species %d has %d rep pts > MAX_REP_PTS=%d",
                                 s, nr, MAX_REP_PTS);
    }

    // ---- Compute global rcut (conservative upper bound for neighbor lists) ----
    // rcut must cover: PLEXCL_RC*(r_i + r_j) + max_rep_offset_p + max_rep_offset_q
    //             and  sqrt(max_dist_sqr_patch) + max_patch_offset_p + max_patch_offset_q
    number rcut = 0.;

    // Max offset of any repulsion point from its particle center
    number max_rep_offset = 0.;
    for (auto &rp : m_RepulsionPoints) {
        number len = std::get<REPULSION_COORDS>(rp).module();
        if (len > max_rep_offset) max_rep_offset = len;
    }
    // Max sum of repulsion radii
    number max_rsum = 0.;
    for (auto &rp : m_RepulsionPoints) {
        number r = std::get<REPULSION_DIST>(rp);
        if (r > max_rsum) max_rsum = r;
    }
    // max_rsum here is a single radius; rcut for repulsion = PLEXCL_RC*(2*max_r) + 2*max_offset
    rcut = std::max(rcut, (number)(PLEXCL_RC * 2 * max_rsum + 2 * max_rep_offset));

    // Max offset of any patch from its particle center
    number max_patch_offset = 0.;
    for (auto &pt : m_PatchesTypes) {
        number len = std::get<PPATCH_POS>(pt).module();
        if (len > max_patch_offset) max_patch_offset = len;
    }
    // Max patch-patch cutoff distance
    number max_patch_dist = 0.;
    for (auto &kv : m_PatchPatchInteractions) {
        number d = sqrt(std::get<PP_MAX_DIST_SQR>(kv.second));
        if (d > max_patch_dist) max_patch_dist = d;
    }
    rcut = std::max(rcut, max_patch_dist + 2 * max_patch_offset);

    if (rcut <= 0.)
        throw oxDNAException("CUDARaspberryInteraction: computed rcut is zero or negative. "
                             "Check topology file.");

    _rcut     = (c_number)rcut;
    _sqr_rcut = _rcut * _rcut;

    // ---- Upload scalar constants ----
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N,         &N,          sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_species, &N_species,  sizeof(int)));
    COPY_NUMBER_TO_FLOAT(MD_sqr_rcut, _sqr_rcut);

    bool angmod_host = patchy_angmod;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_angmod, &angmod_host, sizeof(bool)));

    // NarrowType parameters
    float nt0 = (float)narrow_type.t0, nts = (float)narrow_type.ts;
    float ntc = (float)narrow_type.tc, na  = (float)narrow_type.a;
    float nb  = (float)narrow_type.b;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_narrow_t0, &nt0, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_narrow_ts, &nts, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_narrow_tc, &ntc, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_narrow_a,  &na,  sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_narrow_b,  &nb,  sizeof(float)));

    float e_cut_host = (float)_patch_E_cut;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_E_cut, &e_cut_host, sizeof(float)));

    // ---- Per-species patch/repulsion counts and patch type IDs ----
    {
        int h_N_patches[MAX_SPECIES]  = {};
        int h_N_rep_pts[MAX_SPECIES]  = {};
        int h_ptids[MAX_SPECIES][MAX_PATCHES] = {};

        for (int s = 0; s < N_species; s++) {
            auto &ptype = m_ParticleTypes[s];
            h_N_patches[s] = (int)std::get<PTYPE_PATCH_IDS>(ptype).size();
            h_N_rep_pts[s] = (int)std::get<PTYPE_REP_PTS>  (ptype).size();
            for (int pi = 0; pi < h_N_patches[s]; pi++) {
                h_ptids[s][pi] = std::get<PTYPE_PATCH_IDS>(ptype)[pi];
            }
        }

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches,
                                          h_N_patches, sizeof(int)*N_species));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_rep_pts,
                                          h_N_rep_pts, sizeof(int)*N_species));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_type_ids,
                                          h_ptids, sizeof(h_ptids)));
    }

    // ---- Upload patch geometry textures ----
    {
        int sz = MAX_SPECIES * MAX_PATCHES;
        std::vector<float4> h_patch_pos(sz, make_float4(0,0,0,0));
        std::vector<float4> h_patch_ori(sz, make_float4(0,0,0,0));

        for (int s = 0; s < N_species; s++) {
            auto &pids = std::get<PTYPE_PATCH_IDS>(m_ParticleTypes[s]);
            for (int pi = 0; pi < (int)pids.size(); pi++) {
                auto &pt  = m_PatchesTypes[pids[pi]];
                LR_vector pos = std::get<PPATCH_POS>(pt);
                LR_vector ori = std::get<PPATCH_ORI>(pt);
                h_patch_pos[s*MAX_PATCHES + pi] = make_float4((float)pos.x,(float)pos.y,(float)pos.z,0);
                h_patch_ori[s*MAX_PATCHES + pi] = make_float4((float)ori.x,(float)ori.y,(float)ori.z,0);
            }
        }

        CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_patch_pos, sz * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_patch_pos, h_patch_pos.data(), sz*sizeof(float4), cudaMemcpyHostToDevice));
        GpuUtils::init_texture_object(&_tex_patch_pos,
                                      cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat),
                                      _d_patch_pos, sz);

        CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_patch_ori, sz * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_patch_ori, h_patch_ori.data(), sz*sizeof(float4), cudaMemcpyHostToDevice));
        GpuUtils::init_texture_object(&_tex_patch_ori,
                                      cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat),
                                      _d_patch_ori, sz);
    }

    // ---- Upload repulsion-point textures: xyz = body position, w = radius ----
    {
        int sz = MAX_SPECIES * MAX_REP_PTS;
        std::vector<float4> h_rep(sz, make_float4(0,0,0,0));

        for (int s = 0; s < N_species; s++) {
            auto &rids = std::get<PTYPE_REP_PTS>(m_ParticleTypes[s]);
            for (int ri = 0; ri < (int)rids.size(); ri++) {
                auto &rp  = m_RepulsionPoints[rids[ri]];
                LR_vector pos = std::get<REPULSION_COORDS>(rp);
                number    rad = std::get<REPULSION_DIST>  (rp);
                h_rep[s*MAX_REP_PTS + ri] = make_float4((float)pos.x,(float)pos.y,(float)pos.z,(float)rad);
            }
        }

        CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_rep_pts, sz * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_rep_pts, h_rep.data(), sz*sizeof(float4), cudaMemcpyHostToDevice));
        GpuUtils::init_texture_object(&_tex_rep_pts,
                                      cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat),
                                      _d_rep_pts, sz);
    }

    // ---- Upload patch-pair interaction matrix as float4 texture ----
    // Layout: h_pp_int[p_type_id * MAX_PATCH_TYPES + q_type_id] = (eps, alpha_pow, max_dist_sqr, 0)
    {
        int sz = MAX_PATCH_TYPES * MAX_PATCH_TYPES;
        std::vector<float4> h_pp(sz, make_float4(0,0,0,0));

        for (auto &kv : m_PatchPatchInteractions) {
            int pi = kv.first.first;
            int pj = kv.first.second;
            auto &pp = kv.second;
            float eps      = (float)std::get<PP_INT_EPS>      (pp);
            float alphapow = (float)std::get<PP_INT_ALPHA_POW>(pp);
            float maxdsqr  = (float)std::get<PP_MAX_DIST_SQR> (pp);

            if (pi < MAX_PATCH_TYPES && pj < MAX_PATCH_TYPES) {
                h_pp[pi * MAX_PATCH_TYPES + pj] = make_float4(eps, alphapow, maxdsqr, 0.f);
                // Interactions are symmetric: also fill [pj][pi]
                h_pp[pj * MAX_PATCH_TYPES + pi] = make_float4(eps, alphapow, maxdsqr, 0.f);
            }
        }

        CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_pp_int, sz * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpy(_d_pp_int, h_pp.data(), sz*sizeof(float4), cudaMemcpyHostToDevice));
        GpuUtils::init_texture_object(&_tex_pp_int,
                                      cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat),
                                      _d_pp_int, sz);
    }
}

void CUDARaspberryInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss,
                                              GPU_quat *d_orientations,
                                              c_number4 *d_forces, c_number4 *d_torques,
                                              LR_bonds *d_bonds, CUDABox *d_box) {
    int N = CUDABaseInteraction::_N;

    RI_forces
        <<<_launch_cfg.blocks, _launch_cfg.threads_per_block>>>
        (d_poss, d_orientations, d_forces, d_torques,
         lists->d_matrix_neighs, lists->d_number_neighs,
         _tex_patch_pos, _tex_patch_ori, _tex_rep_pts, _tex_pp_int,
         d_box);
    CUT_CHECK_ERROR("RI_forces error");
}