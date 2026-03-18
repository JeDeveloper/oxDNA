//
// Created by josh evans on 3 December 2024.
// Name is temporary (aka, we're going to keep saying we should get around to renaming it, then not do that)
// code will be pilfered (sorry, "adapted") from both romano/src/Interactions/PatchyShapeInteraction (on oxDNA_torsion repo)
// and a little from Lorenzo Rovigatti's interactions
//

#ifndef RASPBERRYINTERACTION_H
#define RASPBERRYINTERACTION_H

#include "Interactions/BaseInteraction.h"
#include <unordered_map>
#include <unordered_set>

// constants from Flavio Romano's PatchyShapeInteraction
// i'm not 100% sure why r and rc aren't 1
#define PLEXCL_S   1.0f
#define PLEXCL_R   0.9053f
#define PLEXCL_B   677.505671539f
// cutoff for repulsive interaction. if r ^ 2 < ((r1+r2) * PLEXCL_RC) ^ 2, will calculate
#define PLEXCL_RC  0.99888f
#define PLEXCL_EPS 2.0f

// todo: make this dynamic or smth!!!! or at least warn when it will cause issues
#define PATCHY_CUTOFF 0.18f

/**
 * struct to hold parameters for 1-axis narrow type modulation
 */
struct NarrowType {
    number t0, ts, tc, a, b;
    number V_mod(number t) const
    {
        number val = 0;
        t -= t0;
        if (t < 0)
            t *= -1;

        if (t < tc) {
            if (t > ts) {
                // smoothing
                val = b * SQR(tc - t);
            } else
                val = 1.f - a * SQR(t);
        }

        return val;
    }

    number V_modDsin(number t)
    {
        number val = 0;
        number m = 1;
        number tt0 = t - t0;
        // this function is a parabola centered in t0. If t < 0 then the value of the function
        // is the same but the value of its derivative has the opposite sign, so m = -1
        if(tt0 < 0) {
            tt0 *= -1;
            m = -1;
        }

        if(tt0 < tc) {
            number sint = sin(t);
            if(tt0 > ts) {
                // smoothing
                val = m * 2 * b * (tt0 - tc) / sint;
            }
            else {
                if(SQR(sint) > 1e-8) val = -m * 2 * a * tt0 / sint;
                else val = -m * 2 * a;
            }
        }

        return val;
    }
};

const NarrowType NARROW_TYPES[] = {
    // narrow type 0
    {
        0., 0.7, 3.10559, 0.46, 0.133855
    },
    // narrow type 1
    {
        0., 0.2555, 1.304631441617743, 3., 0.7306043547966398
    },
    // narrow type 2
    {
    0., 0.2555, 0.782779, 5., 2.42282
    },
    // narrow type 3
    {
        0., 0.17555, 0.4381832920710734, 13., 8.68949241736805
    },
    // narrow type 4
    {
        0., 0.17555, 0.322741, 17.65, 21.0506
    }
};

// hash unordered pairs
struct UnorderedPairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        // Ensure that (a, b) and (b, a) have the same hash value
        int a = std::min(p.first, p.second);
        int b = std::max(p.first, p.second);
        // Use a simple hash combination technique
        return std::hash<int>()(a) ^ (std::hash<int>()(b) << 1);
    }
};

// Custom equality function for unordered pairs
struct UnorderedPairEqual {
    bool operator()(const std::pair<int, int>& p1, const std::pair<int, int>& p2) const {
        // Since the pair is unordered, (a, b) == (b, a)
        return (p1.first == p2.first && p1.second == p2.second) ||
               (p1.first == p2.second && p1.second == p2.first);
    }
};

/**
* @brief interaction between patchy particles with multiple repulsive points
* we aren't going to use classes for particles and patches to better do forward compatibility with CUDA
*/
class RaspberryInteraction : public BaseInteraction {
protected:
    /**
     * Patch type members:
     * - patch unique ID
     * - patch position
     * - patch alignment (a1 vector). expected to be normalized
     * - patch color (used w/ interaction matrix to find interaction strength)
     * - state variable (for future use)
     * - activation variable (for future use)
     * - patch poly-T spacer length (in nucleotides) (for future use)
     * - patch sticky sequence (for future use)
     */
    using Patch = std::tuple<
            int, //
            LR_vector, // position
            LR_vector, // orientation
            int, // color
            float, //strength
            int, //state variable
            int, //activation variable
            number, // polyT (aka sigma)
            std::string // sticky sequence
            >;
#define PPATCH_TYPE_ID 0
#define PPATCH_POS 1
#define PPATCH_ORI 2
#define PPATCH_COLOR 3
#define PPATCH_STRENGTH 4
#define PPATCH_STATE 5
#define PPATCH_INT_DIST 7
    /**
     * get in loser we're doing native multidentate patches
     * this is mostly an organizational thing
     * tbd
     */
    using PatchGroupType = std::tuple<std::vector< int, Patch&>>;

    /**
     * Repulsion point (not implemented yet)
     * - repulsion point position, repulsion distance
     */
    using RepulsionPoint = std::tuple<int, LR_vector, number>;
#define REPULSION_ID 0
#define REPULSION_COORDS 1
#define REPULSION_DIST 2
public:
    using SignalPassingOperation = std::tuple<std::vector<int>, int, number>;
#define PSIGNAL_SOURCE_STATE_VARS 0
#define PSIGNAL_TARGET_STATE_VAR 1
#define PSIGNAL_PROB 2

protected:
    /**
     * - particle type
     * - number of instances
     * - list of patch ids
     * - list of repulsion point ids
     * - state size
     * - signal-passing operations (if any)
     */
    using ParticleType = std::tuple<int,
                                    int,
                                    std::vector<int>,
                                    std::vector<int>,
                                    int,
                                    std::vector<SignalPassingOperation>>;
#define PTYPE_ID  0
#define PTYPE_INST_COUNT 1
#define PTYPE_PATCH_IDS  2
#define PTYPE_REP_PTS 3
#define PTYPE_STATE_SIZE 4
#define PTYPE_SIGNAL_PASSING_OPS 5

    // repulsion points
    std::vector<RepulsionPoint> m_RepulsionPoints;
    // patch types
    std::vector<Patch> m_PatchesTypes;
    // particle types
    std::vector<ParticleType> m_ParticleTypes;
    // should be length _N, each value is a particle type ID in the above m_ParticleTypes
    std::vector<int> m_ParticleList;

    // runtime variables
    std::vector<std::vector<bool>> m_ParticleStates; // todo
public:
    // bonds. this is tricky
    // use this name here for clarity
    using ParticlePatch = std::pair<int,int>;
protected:
    // list of lists
    // each item in the outer list of particles, the inner list is of patches on the particles
    std::vector<std::vector<ParticlePatch>> m_PatchyBonds;

    // patch color interactions

//    std::unordered_map<std::pair<int, int>, number,  UnorderedPairHash, UnorderedPairEqual> m_PatchColorInteractions;

// todo impl option to specify potential version in input file
    // lorenzo version
    // a PatchPatch object describes how two patches interact
    // there is - by design - very redundant, to minimize required calculations at runtime
    // order: sigma_ss, rcut_ss a_part, b_part, eps
//    using PatchPatch = std::tuple<number, number, number, number, number>;
//#define PP_INT_RCUT_SS  0
//#define PP_INT_SIGMA_SS 1
//#define PP_INT_A_PART   2
//#define PP_INT_B_PART   3
//#define PP_INT_EPS      4

    using PatchPatch = std::tuple<number, number, number, number >;
#define PP_INT_EPS          0
#define PP_INT_ALPHA_POW    1
#define PP_MAX_DIST_SQR     2
#define PP_E_CUT            3

    // i've gone back and forth a LOT about how to work these, settled on this method, for now
    // i don't think this hash or equal function are very fast
    // in this case for speed i am using patch type ids as my hash, color should not be discussed
    // outside initialization
    std::unordered_map<std::pair<int,int>, PatchPatch, UnorderedPairHash, UnorderedPairEqual> m_PatchPatchInteractions;

    number m_nPatchyBondEnergyCutoff;
    number m_nDefaultAlpha;
    bool _has_read_top; // flag to avoid double-reading the top file

    bool patchy_angmod; // whether to use angular modulation in patchy interactions
    NarrowType narrow_type; // parameters for angular modulation
    number _patch_E_cut;

    // Repulsion point distance sum cache
    std::unordered_map<std::pair<int, int>, number, UnorderedPairHash, UnorderedPairEqual> m_RepulsionDistSums;
    std::unordered_map<std::pair<int, int>, number, UnorderedPairHash, UnorderedPairEqual> m_RepulsionDistSqrSumMaxs;

    // state change cache
    std::unordered_set<int> m_ParticlesCanChangeState;
    std::vector<std::vector<number>> m_ActivationProbs; // indexed by particle type, then index of operation in particle type's signal passing ops
    number _dt; // the number of steps per oxDNA units, todo: read this from somewhere else, it must be stored *somewhere* else
public:
    RaspberryInteraction();
    virtual ~RaspberryInteraction();

    virtual void get_settings(input_file &inp);
    // initializes constants for the interaction
    virtual void init();
    // allocate particles
    virtual void allocate_particles(std::vector<BaseParticle *> &particles);

    // utility functions
    int numParticles() const;
    LR_vector getParticlePatchPosition(BaseParticle* p, int patch_idx) const;
    LR_vector getParticlePatchAlign(BaseParticle* p, int patch_idx) const;
    LR_vector getParticleInteractionSitePosition(BaseParticle* p, int int_site_idx) const;
    const Patch& getParticlePatchType(BaseParticle* p, int patch_idx) const;
    number get_r_max_sqr(const int &intSite1, const int &intSite2) const;
    number get_r_sum(const int &intSite1, const int &intSite2) const;

    // pair interaction functions
    virtual number pair_interaction(BaseParticle *p,
                                    BaseParticle *q,
                                    bool compute_r = true,
                                    bool update_forces = false);
    virtual number pair_interaction_bonded(BaseParticle *p,
                                           BaseParticle *q,
                                           bool compute_r = true,
                                           bool update_forces = false);
    virtual number pair_interaction_nonbonded(BaseParticle *p,
                                              BaseParticle *q,
                                              bool compute_r = true,
                                              bool update_forces = false);

    virtual void read_topology(int *N_strands, std::vector<BaseParticle *> &particles);
    virtual int get_N_from_topology();
    virtual void check_input_sanity(std::vector<BaseParticle *> &particles);

    int getPatchBondEnergyCutoff() const {
        return m_nPatchyBondEnergyCutoff;
    }

    void begin_energy_computation() override ;

    // interaction functions
    number repulsive_pt_interaction(BaseParticle *p, BaseParticle *q, bool update_forces);

    number patchy_pt_interaction_angmod(BaseParticle *p, BaseParticle *q, bool update_forces);
    number patchy_pt_interaction_noangmod(BaseParticle *p, BaseParticle *q, bool update_forces);

    void readPatchString(const std::string &patch_line);

    // better to pass refs to objects for speed reasons
    bool patches_can_interact(BaseParticle *p, BaseParticle *q,
                              int ppatch_idx, int qpatch_idx) const;
    bool patch_is_active(BaseParticle* p, const Patch& patch_type) const;
    bool patch_types_interact(const Patch &ppatch_type, const Patch &qpatch_type) const;
//    number patch_types_eps(const Patch &ppatch_type, const Patch &qpatch_type) const;


    // methods for handling locking
    bool is_bound_to(int p, int ppatch_idx, int q, int qpatch_idx) const;
    const ParticlePatch& patch_bound_to(BaseParticle* p, int patch_idx) const;
    bool patch_bound(BaseParticle* p, int patch_idx) const;
    void set_bound_to(int p, int ppatch_idx, int q, int qpatch_idx);
    void clear_bound_to(int p, int ppatch_idx);
    const std::vector< std::vector< ParticlePatch>>& getPatchyBonds() const {
        return m_PatchyBonds;
    }
    const std::vector<ParticlePatch>& getBondsFor(int idx) const;

    static number compute_energy(number patch_dist_sqr, number alpha_exp, number &r8b10) ;

    // signal passing stuff
    void readSignalPassingOperation(const std::string& line);
    bool canChangeState(int particleIdx) const;
    number getActivationProb(int particleType, int operationIdx) const;
    const std::vector<bool>& getParticleState(int particleIdx) const;
};

int stateValue(const std::vector<bool>& stateVec, int stateSize);
std::string readLineNoComment(std::istream& inp);
LR_vector parseVector(const std::string& sz);

#endif //RASPBERRYINTERACTION_H
