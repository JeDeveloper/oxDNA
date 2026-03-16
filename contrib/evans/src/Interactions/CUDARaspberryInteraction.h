//
// CUDARaspberryInteraction.h
// Created by josh evans on 3 December 2024.
//
// CUDA implementation of RaspberryInteraction.
// Modelled after CUDADetailedPatchySwapInteraction (contrib/rovigatti).
//

#ifndef CUDARASPBERRYINTERACTION_H
#define CUDARASPBERRYINTERACTION_H

#include "RaspberryInteraction.h"
#include "CUDA/Interactions/CUDABaseInteraction.h"

/**
 * @brief CUDA implementation of RaspberryInteraction.
 *
 * GPU constant-memory layout (per-species arrays use MAX_SPECIES stride,
 * per-patch arrays use MAX_PATCHES stride, etc.):
 *
 *   tex_patch_pos[species * MAX_PATCHES + patch_idx] : float4  xyz = body-frame position
 *   tex_patch_ori[species * MAX_PATCHES + patch_idx] : float4  xyz = body-frame orientation (a1)
 *   tex_rep_pts  [species * MAX_REP_PTS  + rep_idx ] : float4  xyz = body-frame position, w = radius
 *   tex_pp_int   [ptype * MAX_PATCH_TYPES + qtype  ] : float4  (eps, alpha_pow, max_dist_sqr, unused)
 *
 * Patch-type IDs for each (species, patch_index) are in MD_patch_type_ids[][].
 * Note: patch locking is NOT implemented on GPU (same approach as DPS).
 */
class CUDARaspberryInteraction : public CUDABaseInteraction, public RaspberryInteraction {
protected:
    // Per-species patch geometry: body-frame position and orientation
    float4 *_d_patch_pos = nullptr;
    float4 *_d_patch_ori = nullptr;
    cudaTextureObject_t _tex_patch_pos = 0;
    cudaTextureObject_t _tex_patch_ori = 0;

    // Per-species repulsion-point geometry: body-frame position (xyz) and radius (w)
    float4 *_d_rep_pts = nullptr;
    cudaTextureObject_t _tex_rep_pts = 0;

    // Patch-pair interaction matrix: (eps, alpha_pow, max_dist_sqr, unused)
    // Indexed flat as [p_patch_type_id * MAX_PATCH_TYPES + q_patch_type_id]
    float4 *_d_pp_int = nullptr;
    cudaTextureObject_t _tex_pp_int = 0;

public:
    static const int MAX_PATCHES     = 20;
    static const int MAX_REP_PTS     = 30;
    static const int MAX_SPECIES     = 16;
    static const int MAX_PATCH_TYPES = 64;

    CUDARaspberryInteraction();
    virtual ~CUDARaspberryInteraction();

    void get_settings(input_file &inp);
    void cuda_init(int N);

    c_number get_cuda_rcut() {
        return this->_rcut;
    }

    void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations,
                        c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds,
                        CUDABox *d_box);
};

extern "C" BaseInteraction *make_CUDARaspberryPatchyInteraction() {
    return new CUDARaspberryInteraction();
}

#endif // CUDARASPBERRYINTERACTION_H