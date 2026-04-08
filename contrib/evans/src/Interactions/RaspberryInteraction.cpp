//
// Created by josh evans on 3 December 2024
// Name is temporary (aka, we're going to keep saying we should get around to renaming it, then not do that)
// code will be pilfered (sorry, "adapted") from both romano/src/Interactions/PatchyShapeInteraction (on oxDNA_torsion repo)
// and a little from Lorenzo Rovigatti's interactions
//

#include <sstream>
#include "RaspberryInteraction.h"

#include <regex>

#include "Particles/PatchyParticle.h"
#include "../Particles/RaspberryParticle.h"

// todo
//#define PATCHY_CUTOFF 0.18f
#define MIN_E           0.0001

RaspberryInteraction::RaspberryInteraction()  : BaseInteraction(){
    // we actually don't want to add interactions to map here, since for raspberry particles
    // this will depend somewhat on inputs
    m_nPatchyBondEnergyCutoff = -0.1;
    m_nDefaultAlpha = 0;
    patchy_angmod = true;
    _has_read_top = false;
    narrow_type = NARROW_TYPES[0];
}

RaspberryInteraction::~RaspberryInteraction() = default;

/**
 * function to allocate particles
 */
void RaspberryInteraction::init() {
    number r8b10 = powf(PATCHY_CUTOFF, 8.f) / powf(m_nDefaultAlpha, 10.f);
    _patch_E_cut = -1.001f * expf(-0.5f * r8b10 * SQR(PATCHY_CUTOFF));
    OX_LOG(Logger::LOG_INFO, "INFO: setting _patch_E_cut to %f which is %f, with alpha=%f", _patch_E_cut,-1.001f * expf(-0.5f * r8b10 * SQR(PATCHY_CUTOFF)), m_nDefaultAlpha);
}

void RaspberryInteraction::get_settings(input_file &inp) {
    BaseInteraction::get_settings(inp);
    getInputNumber(&inp, "dt", &_dt, 1);

    getInputNumber(&inp, "PATCHY_bond_energy", &m_nPatchyBondEnergyCutoff, 0);
    // default alpha value if it isn't specified
    getInputNumber(&inp, "PATCHY_alpha", &m_nDefaultAlpha, 0);
    getInputBool(&inp, "patchy_angmod", &patchy_angmod, 0);
    if (patchy_angmod){
        int narrow_type_num;
        if (getInputInt(&inp, "narrow_type", &narrow_type_num, 0) == KEY_FOUND) {
            if (narrow_type_num < 0 || narrow_type_num > 4) {
                throw oxDNAException("Invalid narrow type specified: %d", narrow_type_num);
            }
            narrow_type = NARROW_TYPES[narrow_type_num];
        }
        else {
            OX_LOG(Logger::LOG_INFO, "No narrow type enumeration specified, will look for explicit numeric narrow type parameters.");
            narrow_type.t0 = 0.;
            if (getInputNumber(&inp, "PATCHY_theta_t0", &narrow_type.t0, 0) != KEY_FOUND) {
                OX_LOG(Logger::LOG_INFO, "No angmod theta t0 specified, defaulting to 0");
            }
            getInputNumber(&inp, "PATCHY_theta_ts", &narrow_type.ts, 1);
            getInputNumber(&inp, "PATCHY_theta_tc", &narrow_type.tc, 1);
            getInputNumber(&inp, "PATCHY_theta_a", &narrow_type.a, 1);
            getInputNumber(&inp, "PATCHY_theta_b", &narrow_type.b, 1);
        }
    }
}


void RaspberryInteraction::allocate_particles(std::vector<BaseParticle *> &particles) {
    m_ParticleStates.resize(particles.size());
    m_PatchyBonds.resize(particles.size());
    int i_type = 0;
    int type_count = 0;
    for (int i = 0; i < particles.size(); i++){
        // assign particle type
        m_ParticleList[i] = i_type;

        // ordering of interaction centers is important!
        // first we do patches, as pairs of position, a1
        // then repuslive spheres!
        const ParticleType& particleType = m_ParticleTypes[i_type];
        int num_int_centers = std::get<PTYPE_REP_PTS>(particleType).size()
                + 2 * std::get<PTYPE_PATCH_IDS>(particleType).size();
        // create & populate interaction center list to use in particle object constructor
        std::vector<LR_vector> int_centers(num_int_centers);
        // use same indexer for repulsion points & patches
        int j = 0;

        for (; j < 2*std::get<PTYPE_PATCH_IDS>(particleType).size(); j += 2){
            // two "int centers" for each patch: position and a1
            int idx = j /2 ;
            assert(idx < std::get<PTYPE_PATCH_IDS>(particleType).size());
            int iPatch = std::get<PTYPE_PATCH_IDS>(particleType)[idx];
            assert(-1 < iPatch && iPatch < m_PatchesTypes.size());
            // assign position from patch info
            int_centers[j] = std::get<PPATCH_POS>(m_PatchesTypes[iPatch]);
            // assign a1 from patch info
            int_centers[j+1] = std::get<PPATCH_ORI>(m_PatchesTypes[iPatch]);
        }

        for (; j < num_int_centers; j++){
            // get index of info in m_RepulsionPoints
            int idx = j - 2 * std::get<PTYPE_PATCH_IDS>(particleType).size();
            int iRepulsionPoint = std::get<PTYPE_REP_PTS>(particleType)[idx];
            assert(iRepulsionPoint < m_RepulsionPoints.size());
            // set int center to repulsion point position
            int_centers[j] = std::get<1>(m_RepulsionPoints[iRepulsionPoint]);
        }


        particles[i] = new RaspberryParticle(std::move(int_centers));
        particles[i]->index = i;
        particles[i]->strand_id = i;
        particles[i]->type = i_type;
        particles[i]->btype = i_type; // needed for CUDA: btype is packed into ppos.w

        // init interaction runtime variables
        // particle states
        // todo: at some point we will need to be able to load these from somewhere
        m_ParticleStates[i].resize(std::get<PTYPE_STATE_SIZE>(particleType));
        m_ParticleStates[i][0] = true;
        for (int ii = 1; ii < m_ParticleStates[i].size(); ii++){
            m_ParticleStates[i][ii] = false;
        }
        assert(i < m_PatchyBonds.size());
        m_PatchyBonds[i].resize(std::get<PTYPE_PATCH_IDS>(particleType).size());
        // for each patch
        for (int ii = 0; ii < m_PatchyBonds[i].size(); ii++){
            // ParticlePatch = {-1, -1} means no bond
            assert(ii < m_PatchyBonds[i].size());
            m_PatchyBonds[i][ii] = {-1, -1};
        }

        type_count++;
        if (type_count == std::get<PTYPE_INST_COUNT>(particleType)){
            i_type++;
            type_count = 0;
        }
    }
}

int RaspberryInteraction::numParticles() const {
    return m_ParticleList.size();
}

const RaspberryInteraction::Patch& RaspberryInteraction::getParticlePatchType(BaseParticle *p, int patch_idx) const{
    const ParticleType& p_type = m_ParticleTypes[p->type];
    const int ppatch_tid = std::get<PTYPE_PATCH_IDS>(p_type)[patch_idx];
    return m_PatchesTypes[ppatch_tid];
}

// interaction sites in particle positions

/***
 * retrieves the position of patch with the specified index on the specified particle
 * @param p
 * @param patch_idx INDEX of the patch WITHIN THE PARTICLE TYPE
 * @return
 */
LR_vector RaspberryInteraction::getParticlePatchPosition(BaseParticle *p, int patch_idx) const {
    // patch positions are listed first
    return p->int_centers[2*patch_idx];
}

/**
 * retrieves the alignment vector of patch with the specified position on the specified particle
 * @param p
 * @param patch_idx INDEX of the patch WITHIN THE PARTICLE TYPE
 * @return
 */
LR_vector RaspberryInteraction::getParticlePatchAlign(BaseParticle *p, int patch_idx) const {
    const ParticleType* particleType = &m_ParticleTypes[m_ParticleList[p->get_index()]];
    int idx = patch_idx + std::get<PTYPE_PATCH_IDS>(*particleType).size();
    // patch orientations are listed after patch positions
    return p->int_centers[2*patch_idx + 1];
}

/**
 *
 * @param p
 * @param int_site_idx INDEX of the interaction sitw WITHIN THE PARTICLE TYPE
 * @return
 */
LR_vector RaspberryInteraction::getParticleInteractionSitePosition(BaseParticle *p, int int_site_idx) const {
    const ParticleType* particleType = &m_ParticleTypes[m_ParticleList[p->get_index()]];
    // interaction sites are listed after the patch geometry
    int idx = int_site_idx + 2 * std::get<PTYPE_PATCH_IDS>(*particleType).size();
    assert(idx < p->int_centers.size());
    return p->int_centers[idx];
}

number RaspberryInteraction::pair_interaction(BaseParticle *p,
                                              BaseParticle *q,
                                              bool compute_r,
                                              bool update_forces) {
    return pair_interaction_nonbonded(p, q, compute_r, update_forces);
}

number RaspberryInteraction::pair_interaction_bonded(BaseParticle *p,
                                                     BaseParticle *q,
                                                     bool compute_r,
                                                     bool update_forces) {
    // going off Lorenzo's example we are treating all particles as nonbonded
    return 0.;
}

number RaspberryInteraction::pair_interaction_nonbonded(BaseParticle *p,
                                                        BaseParticle *q,
                                                        bool compute_r,
                                                        bool update_forces) {
    number e = 0.;
    if(compute_r) {
        _computed_r = _box->min_image(p->pos, q->pos);
    }

    e += repulsive_pt_interaction(p, q, update_forces);
    if (patchy_angmod) {
        e += patchy_pt_interaction_angmod(p, q, update_forces);
    }
    else {
        e += patchy_pt_interaction_noangmod(p, q, update_forces);
    }

    return e;
}

int RaspberryInteraction::get_N_from_topology() {
    std::ifstream topology(_topology_filename, std::ios::in);
    if(!topology.good()) throw oxDNAException("Can't read topology file '%s'. Aborting", _topology_filename);
    std::string sz, _;
    int N = 0;
    int n;

    // topology file should only be a few dozen lines, we can just skim it
    while (std::getline(topology, sz)) {
        if (sz.length() > 2 && sz.substr(0,2) == "iC"){
            // yuck way to do this but i don't care
            std::stringstream ss(sz);
            ss >> _ >> _ >> n;
            N += n;
        }
    }
    return N;
}

/**
 * reads the topology file
 * @param N_strands
 * @param particles
 */
void RaspberryInteraction::read_topology(int *N_strands, std::vector<BaseParticle *> &particles) {
    if (!_has_read_top) {
        // open topology file
        std::ifstream topology(_topology_filename, std::ios::in);
        if (!topology.good()) throw oxDNAException("Can't read topology file '%s'. Aborting", _topology_filename);

        // resize list of particle type IDs to match particle pointers
        m_ParticleList.resize(particles.size());
        // set number of "strands"
        *N_strands = m_ParticleList.size();
        // read header
        std::string first_line = readLineNoComment(topology);
        // skip first line entirely (for now)

        // let's read all lines first
        std::vector<std::string> particle_type_lines;
        std::vector<std::string> patch_type_lines;
        std::vector<std::string> patch_group_lines;
        std::vector<std::string> repulsion_pt_lines;
        std::vector<std::string> signal_passing_operations;
        std::vector<std::string> patch_interactions;

        std::string sz;
        while (std::getline(topology, sz)) {
            // read line
            sz = Utils::trim(sz);
            // if line is not blank or a comment
            if (sz.length() > 0 && sz[0] != '#') {
                // if line is patch descriptor
                if (sz.substr(0, 2) == "iP") {
                    patch_type_lines.push_back(sz);
                }
                    // particle type ("corpuscule") descriptor
                else if (sz.substr(0, 2) == "iC") {
                    particle_type_lines.push_back(sz);
                }
                    // operation descriptor
                else if (sz.substr(0, 2) == "iO") {
                    signal_passing_operations.push_back(sz);
                } else if (sz.substr(0, 2) == "iG") {
                    patch_group_lines.push_back(sz);
                } else if (sz.substr(0, 2) == "iR") {
                    repulsion_pt_lines.push_back(sz);
                } else if (sz.substr(0, 2) == "iI") {
                    patch_interactions.push_back(sz);
                }
                else {
                    throw oxDNAException("Malformed topology! Line `" + sz + "` does not clearly convey information");
                }
            }
        }
        // resize patch type vector
        m_PatchesTypes.resize(patch_type_lines.size());
        // read patches
        for (int i = 0; i < patch_type_lines.size(); i++) {
            readPatchString(patch_type_lines[i]);
        }

        // read patch groups
        for (int i = 0; i < patch_group_lines.size(); i++) {
            std::stringstream ss(patch_group_lines[i].substr(2, patch_group_lines[i].size() - 2));
            std::vector<int> patches;
            int p;
            while (ss >> p) {
                patches.push_back(p);
            }
            // todo: if patch groups do anything put that here
        }

        // read repulsion points
        m_RepulsionPoints.resize(repulsion_pt_lines.size());
        for (int i = 0; i < repulsion_pt_lines.size(); i++) {
            std::stringstream ss(repulsion_pt_lines[i].substr(2, repulsion_pt_lines[i].size() - 2));
            LR_vector position;
            number r;
            std::string pos_str;
            if (!(ss >> pos_str >> r)) {
                throw oxDNAException("Invalid repulsion point type str `" + repulsion_pt_lines[i] + "`!");
            }
            try {
                position = parseVector(pos_str);
            } catch (oxDNAException &e) {
                throw oxDNAException("Invalid repulsion point type str `" + repulsion_pt_lines[i] + "`!");
            }
            m_RepulsionPoints[i] = {i, position, r};
        }
        // cache sums of all possible repulsion point pairs
        // todo: warn if this is too big?
        for (int i = 0; i < m_RepulsionPoints.size(); i++){
            for (int j = i; j < m_RepulsionPoints.size(); j++){
                m_RepulsionDistSums[{i, j}] = std::get<REPULSION_DIST>(m_RepulsionPoints[i]) + std::get<REPULSION_DIST>(m_RepulsionPoints[j]);
                m_RepulsionDistSqrSumMaxs[{i, j}] = 1.2 * SQR( (m_RepulsionDistSums[{i, j}]) );
            }
        }

        // read particle types
        m_ParticleTypes.resize(particle_type_lines.size());
        for (int i = 0; i < particle_type_lines.size(); i++) {
            std::stringstream ss(particle_type_lines[i].substr(2, particle_type_lines[i].size() - 2));
            int iParticleType;
            std::string patch_id_strs, interaction_pt_id_strs;
            if (!(ss >> iParticleType >> std::get<PTYPE_INST_COUNT>(m_ParticleTypes[iParticleType]) >> patch_id_strs)) {
                throw oxDNAException("Invalid particle type str `" + particle_type_lines[i] + "`!");
            }
            if (!(ss >> interaction_pt_id_strs)) {
                OX_LOG(Logger::LOG_WARNING, "Particle type str `%s` does not specify interaction points! Defaulting to none.", particle_type_lines[i].c_str());
                interaction_pt_id_strs = "";
            }
            if (iParticleType >= m_ParticleTypes.size()) {
                throw oxDNAException("Invalid particle type ID %d", iParticleType);
            }
            // assign particle type
            std::get<PTYPE_ID>(m_ParticleTypes[iParticleType]) = iParticleType;
            std::vector<std::string> patch_id_strs_list = Utils::split(patch_id_strs, ',');
            std::vector<std::string> int_pt_strs_list = Utils::split(interaction_pt_id_strs, ',');
            // process patch IDs
            for (std::string &sz: patch_id_strs_list) {
                int patch = std::stoi(sz);
                if (patch >  -1) { // skip "dummy" patch -1
                    assert(patch < static_cast<int>(m_PatchesTypes.size()));
                    std::get<PTYPE_PATCH_IDS>(m_ParticleTypes[iParticleType]).push_back(patch);
                }
            }
            // process interaction points
            for (std::string &sz: int_pt_strs_list) {
                int repulsionPt = std::stoi(sz);
                assert(repulsionPt < m_RepulsionPoints.size());
                std::get<PTYPE_REP_PTS>(m_ParticleTypes[iParticleType]).push_back(repulsionPt);
            }
            // TODO: operations
            int iStateSize;
            // state size is optional
            if (ss >> iStateSize) {
                std::get<PTYPE_STATE_SIZE>(m_ParticleTypes[iParticleType]) = iStateSize;
            } else {
                std::get<PTYPE_STATE_SIZE>(m_ParticleTypes[iParticleType]) = 1;
            }
        }

        // todo: option to use the Arrhenaeus equation so the probability can change dynamaically with T
        for (const auto& signal_str: signal_passing_operations) {
            readSignalPassingOperation(signal_str);
        }
        // compute signal passing probabilities
        m_ActivationProbs.resize(m_ParticleList.size());
        for (int iParticleType = 0; iParticleType < m_ParticleTypes.size(); iParticleType++) {
            int num_ops = std::get<PTYPE_SIGNAL_PASSING_OPS>(m_ParticleTypes[iParticleType]).size();
            m_ActivationProbs[iParticleType].resize(num_ops);
            for (int iOpIdx = 0; iOpIdx < num_ops; iOpIdx++) {
                // probabilty is written in the top file as the probabilty per oxDNA time unit
                number base_rate = std::get<PSIGNAL_PROB>(std::get<PTYPE_SIGNAL_PASSING_OPS>(m_ParticleTypes[iParticleType])[iOpIdx]);
                if (base_rate == 1) {
                    m_ActivationProbs[iParticleType][iOpIdx] = 1.;
                }
                else {
                    // convert from probability per time unit to probability per time step
                    m_ActivationProbs[iParticleType][iOpIdx] = 1 - (log(1. - base_rate) / -log(_dt));
                    assert(m_ActivationProbs[iParticleType][iOpIdx] > 0);
                    assert(m_ActivationProbs[iParticleType][iOpIdx] < 1.);
                }
            }
        }

        // close topology file
        topology.close();
        // TODO: CUDA version of this func will need to cache vs. max patches
        std::vector<std::tuple<int, int, number>> patch_interaction_inputs;
        if (patch_interactions.empty()) {
            // define color interactions
            for (int pi = 0; pi < m_PatchesTypes.size(); pi++) {
                int pi_color = std::get<PPATCH_COLOR>(m_PatchesTypes[pi]);
                for (int pj = 0; pj < m_PatchesTypes.size(); pj++) {
                    int pj_color = std::get<PPATCH_COLOR>(m_PatchesTypes[pj]);
                    // use petr + flavio version of default color interactions
                    // number pj_dist = std::get<PPATCH_INT_DIST>(m_PatchesTypes[pj]);
                    if (((abs(pi_color) >= 20) && (pi_color + pj_color == 0)) ||
                        ((abs(pi_color) < 20) && (pi_color == pj_color))) {
                        patch_interaction_inputs.emplace_back(pi, pj, 1.);
                    }
                }
            }
        }
        else {
            int color1, color2;
            for (auto &interaction_string : patch_interactions) {
                std::stringstream ss(interaction_string.substr(2, interaction_string.size() - 2));
                number strength = 1;
                if (!(ss >> color1 >> color2)) {
                    throw oxDNAException("Invalid patch interaction str `" + interaction_string + "`! Does not have two colors");
                }
                if (!(ss >> strength)) {
                    OX_LOG(Logger::LOG_WARNING, "Patch interaction `%s` does not specify strength! Defaulting to 1.", interaction_string.c_str());
                }
                // I don't care how inefficient this is, it only runs once per simulation
                // find all patches with these colors
                for (int pi = 0; pi < m_PatchesTypes.size(); pi++) {
                    int pi_color = std::get<PPATCH_COLOR>(m_PatchesTypes[pi]);
                    if (pi_color == color1) {
                        for (int pj = 0; pj < m_PatchesTypes.size(); pj++) {
                            int pj_color = std::get<PPATCH_COLOR>(m_PatchesTypes[pj]);
                            if (pj_color == color2) {
                                patch_interaction_inputs.emplace_back(pi, pj, strength);
                            }
                        }
                    }
                }
            }
        }
        for (auto [pi, pj, strength] : patch_interaction_inputs) {
            // temporary variable
            number r8b10;
            number dist_sum = std::get<PPATCH_INT_DIST>(m_PatchesTypes[pi]) + std::get<PPATCH_INT_DIST>(m_PatchesTypes[pj]);
            // find interaction strength by computing the geometric mean of the two patch strengths
            number interaction_strength = strength * sqrt(std::get<PPATCH_STRENGTH>(m_PatchesTypes[pi]) * std::get<PPATCH_STRENGTH>(m_PatchesTypes[pj]));
            number dist_cutoff_sqrd = pow(dist_sum, 2) * pow(
                    log(1.001) - log(MIN_E),
                    0.2
                    );
            // check energy at expected distance
            number e_cutoff = compute_energy(dist_cutoff_sqrd, pow(dist_sum, 10), r8b10);
            m_PatchPatchInteractions[{pi, pj}] = {
                interaction_strength,
                pow(dist_sum, 10),
                dist_cutoff_sqrd,
                e_cutoff // probably like MIN_E
            };
        }
        _has_read_top = true;
    }
    allocate_particles(particles);
    // rcut
    _rcut = 0.; // start from 0, we are calculating max value
    // patch interaction maximum distances
    for (int i = 0; i < m_PatchesTypes.size(); i++) {
        for (int j = 0; j < m_PatchesTypes.size(); j++) {
            if (patch_types_interact(m_PatchesTypes[i], m_PatchesTypes[j])) {
                number dist_sum = sqrt(std::get<PP_MAX_DIST_SQR>(m_PatchPatchInteractions[{i, j}]));
                dist_sum += sqrt(std::get<PPATCH_POS>(m_PatchesTypes[i]).norm());
                dist_sum += sqrt(std::get<PPATCH_POS>(m_PatchesTypes[j]).norm());
                if (dist_sum > _rcut) {
                    _rcut = dist_sum;
                }
            }
        }
    }
    // max repulsive interaction distances
    for (int i = 0; i < m_RepulsionPoints.size(); i++) {
        for (int j = 0; j < m_RepulsionPoints.size(); j++) {
            // max possible repulsion dist =
            // max distance between the repulsion points + the sum of the distances of the points from
            // the particle center
            number r = sqrt(m_RepulsionDistSqrSumMaxs[{i, j}]);
            r += sqrt(std::get<REPULSION_COORDS>(m_RepulsionPoints[i]).norm());
            r += sqrt(std::get<REPULSION_COORDS>(m_RepulsionPoints[j]).norm());
            if (r > _rcut) {
                _rcut = r;
            }
        }
    }

    _sqr_rcut = SQR(_rcut);
}

void RaspberryInteraction::readPatchString(const std::string& patch_line) {
    std::string patch_infostr = patch_line.substr(3);
    std::stringstream ss(patch_infostr);
    int iTypeID, iColor, iState, iActivation;
    number fStrength, fPatchDist;
    std::string vecStr, oriStr;
    LR_vector position, orientation;

    // order of line should be UID color strength position a1 statevar activationvar

    // read patch type ID
    if (!(ss >> iTypeID)){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }

    if (!(ss >> fStrength)){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }
    if (fStrength != 1.0){
        OX_LOG(Logger::LOG_WARNING, "Patch strength is only used if you don't explicitly specify the interaction matrix");
    }

    // read color
    if (!(ss >> iColor)){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }

    // read patch position
    if (!(ss >> vecStr)){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }
    try {
        position = parseVector(vecStr);
    } catch (oxDNAException &e){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }

    // read patch orientation
    if (!(ss >> oriStr)){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }
    try {
        orientation = parseVector(oriStr);
    } catch (oxDNAException &e){
        throw oxDNAException("Invalid patch type str `" + patch_line + "`!");
    }

    // patch distance
    if (!(ss >> fPatchDist) || (fPatchDist == 0)) {
        if (m_nDefaultAlpha > 0){
            fPatchDist = m_nDefaultAlpha;
        }
        else {
            throw oxDNAException("Interaction distance not specified (generic) in input file or (specific) in `" + patch_line + "`!");
        }
    }
    assert(fPatchDist > 0);

    // following values are optional!
    if (!(ss >> iState && ss >> iActivation)){
        iState = 0;
        iActivation =0;
    }
    if (iTypeID >= m_PatchesTypes.size()){
        throw oxDNAException("Invalid patch type ID %d", iTypeID);
    }
    // TODO: polyTs, sequence?
    m_PatchesTypes[iTypeID] = {iTypeID,
                               position,
                               orientation,
                               iColor,
                               fStrength,
                               iState,
                               iActivation,
                               fPatchDist,
                               ""};
}

void RaspberryInteraction::readSignalPassingOperation(const std::string &signal_str) {
    std::stringstream ss(signal_str);
    int iParticleType;
    std::vector<int> iOriginStateVars;
    int iTargetStateVar;
    number fActivationProb;
    if (!(ss >> iParticleType)) {
        throw oxDNAException("Invalid signal passing operation str `%s`! Does not have particle type", signal_str.c_str());
    }
    if ((iParticleType >= m_ParticleTypes.size()) || (iParticleType < 0)) {
        throw oxDNAException("Invalid particle type ID %d in signal passing operation %s", iParticleType, signal_str.c_str());
    }
    std::string originStateVarsStr;
    if (!(ss >> originStateVarsStr)) {
        throw oxDNAException("Invalid signal passing operation str `%s`! Does not have origin state vars", signal_str.c_str());
    }
    // Delimiter pattern
    std::regex del(",");

    // regex iterator
    std::sregex_token_iterator it(originStateVarsStr.begin(),
                     originStateVarsStr.end(), del, -1);

    // End iterator for the
    std::sregex_token_iterator end;

    while (it != end) {
        try {
            iOriginStateVars.push_back(std::stoi(*it));
        }
        catch (const std::invalid_argument &e) {
            throw oxDNAException("Invalid signal passing operation str `%s`! Origin state vars should be comma-separated integers", signal_str.c_str());
        }
        ++it;
    }
    if (!(ss >> iTargetStateVar)) {
        throw oxDNAException("Invalid signal passing operation str `%s`! Does not have target state var", signal_str.c_str());
    }
    if (!(ss >> fActivationProb)) {
        OX_LOG(Logger::LOG_INFO, "No activation probability specified in signal passing operation str `%s`, defaulting to 1", signal_str.c_str());
        fActivationProb = 1.;
    }
    std::get<PTYPE_SIGNAL_PASSING_OPS>(m_ParticleTypes[iParticleType]).push_back({iOriginStateVars, iTargetStateVar, fActivationProb});
}

void RaspberryInteraction::check_input_sanity(std::vector<BaseParticle *> &particles) {
    // The integrator assumes an isotropic moment of inertia (I_tensor = I * identity).
    // Warn per particle type if its repulsion site geometry renders this assumption tenuous.
    for (int iType = 0; iType < (int)m_ParticleTypes.size(); iType++) {
        const auto& particleType = m_ParticleTypes[iType];
        const auto& repPtIds = std::get<PTYPE_REP_PTS>(particleType);
        if ((int)repPtIds.size() < 2) continue;

        // Check 1: center of mass of repulsion sites should be at the particle origin.
        // If not, the effective rotation center is offset from the interaction geometry center.
        LR_vector com(0, 0, 0);
        for (int id : repPtIds) {
            com += std::get<REPULSION_COORDS>(m_RepulsionPoints[id]);
        }
        com = com / (number)repPtIds.size();
        if (com.norm() > 1e-4) {
            OX_LOG(Logger::LOG_WARNING,
                "Particle type %d: repulsion site center of mass is offset from the particle origin by %g. "
                "Rotational dynamics may be incorrect.",
                iType, sqrt(com.norm()));
        }

        // Check 2: inertia tensor anisotropy.
        // Compute the inertia tensor treating each repulsion site as a unit-mass point.
        // For isotropy we need I_tensor = lambda * identity (all eigenvalues equal, off-diagonals zero).
        number Ixx = 0, Iyy = 0, Izz = 0;
        number Ixy = 0, Ixz = 0, Iyz = 0;
        for (int id : repPtIds) {
            const LR_vector& r = std::get<REPULSION_COORDS>(m_RepulsionPoints[id]);
            Ixx += r.y*r.y + r.z*r.z;
            Iyy += r.x*r.x + r.z*r.z;
            Izz += r.x*r.x + r.y*r.y;
            Ixy -= r.x*r.y;
            Ixz -= r.x*r.z;
            Iyz -= r.y*r.z;
        }

        number lambda = (Ixx + Iyy + Izz) / 3.;
        if (lambda < 1e-10) continue;

        // Frobenius norm of (I_tensor - lambda * identity).
        // Off-diagonal terms appear twice in the symmetric matrix.
        number frob_sq = SQR(Ixx - lambda) + SQR(Iyy - lambda) + SQR(Izz - lambda)
                       + 2.*SQR(Ixy) + 2.*SQR(Ixz) + 2.*SQR(Iyz);
        // Normalise by lambda so the metric is dimensionless and scale-independent.
        number relative_anisotropy = sqrt(frob_sq) / lambda;

        if (relative_anisotropy > 0.1) {
            OX_LOG(Logger::LOG_WARNING,
                "Particle type %d: repulsion site distribution is significantly anisotropic "
                "(relative anisotropy = %.3f, threshold = 0.1). "
                "The isotropic mass distribution assumption used by the integrator may not hold. "
                "Consider redistributing repulsion sites more symmetrically.",
                iType, relative_anisotropy);
        }
    }

//    printf("###########\n");
//    BaseParticle* p = CONFIG_INFO->particles()[0];
//    BaseParticle* q = CONFIG_INFO->particles()[1];
//
//    p->pos = LR_vector(0,0,0);
//
//    for (double pp_dist = 1; pp_dist < 1.125 ; pp_dist+=0.001)
//    {
//        _computed_r = q->pos = LR_vector(pp_dist, 0, 0);
//        p->force = {0,0,0};
//        double e = repulsive_pt_interaction(p, q, true);
//        double force = sqrt(p->force.norm());
//        printf("%f %f\n", pp_dist, force);
//    }
}

void RaspberryInteraction::begin_energy_computation() {
    // this is the place to update particle states
    // compared to the particle-particle computations this is relatively computationally light
    // there may be a faster way to go through these but for now we'll just check every possible operation on every particle, and optimize later if needed
    for (int iParticleIdx = 0; iParticleIdx < m_ParticleList.size(); iParticleIdx++) {
        if (canChangeState(iParticleIdx)) {
            int particleTypeIdx = m_ParticleList[iParticleIdx];
            RaspberryParticle* pp = static_cast<RaspberryParticle*>(CONFIG_INFO->particles()[iParticleIdx]);
            const ParticleType& particleType = m_ParticleTypes[m_ParticleList[iParticleIdx]];
            // a particle with no state variables can't have any operations, so skip it
            // can assume if a particle has state variables, it has operations
            for (int iOpIdx = 0; iOpIdx < std::get<PTYPE_SIGNAL_PASSING_OPS>(particleType).size(); iOpIdx++) {
                SignalPassingOperation signal_op = std::get<PTYPE_SIGNAL_PASSING_OPS>(particleType)[iOpIdx];
                // check if all origin state vars are active
                bool all_active = true;
                for (int origin_var: std::get<PSIGNAL_SOURCE_STATE_VARS>(signal_op)) {
                    // todo: remove this assert once I'm comfortable
                    assert(origin_var >= 0 && origin_var < m_ParticleStates[iParticleIdx].size());
                    if (!m_ParticleStates[iParticleIdx][origin_var]) {
                        all_active = false;
                        break;
                    }
                }
                if (all_active) {
                    number activationProb = getActivationProb(particleTypeIdx, iOpIdx);
                    if (drand48() < activationProb) {
                        // activate target state var
                        int target_var = std::get<PSIGNAL_TARGET_STATE_VAR>(signal_op);
                        assert(target_var >= 0 && target_var < m_ParticleStates[iParticleIdx].size());
                        m_ParticleStates[iParticleIdx][target_var] = true;
                        OX_LOG(Logger::LOG_DEBUG, "Activating state var %d on particle %d of type %d with operation %d (prob %f)", target_var, iParticleIdx, particleTypeIdx, iOpIdx, activationProb);
                    }
                }
            }
            // update whether particle can change state in the future (if it has any operations left that could be activated)
            bool canChangeStateAgain = false;
            for (int iState = 0; iState < std::get<PTYPE_STATE_SIZE>(m_ParticleTypes[particleTypeIdx]); iState++) {
                if (!m_ParticleStates[iParticleIdx][iState]) {
                    canChangeStateAgain = true;
                    break;
                }
            }
            if (!canChangeStateAgain) {
                assert(m_ParticlesCanChangeState.erase(iParticleIdx) == 1);
            }
        }
    }
}

number RaspberryInteraction::getActivationProb(int particleType, int operationIdx) const {
    assert(particleType >= 0 && particleType < m_ParticleTypes.size());
    assert(operationIdx >= 0 && operationIdx < std::get<PTYPE_SIGNAL_PASSING_OPS>(m_ParticleTypes[particleType]).size());
    assert(std::get<PTYPE_SIGNAL_PASSING_OPS>(m_ParticleTypes[particleType]).size() == m_ActivationProbs[particleType].size());
    return m_ActivationProbs[particleType][operationIdx];
}

/**
 * a lot of this is pilfered from DetailedPatchySwapInteraction::_patchy_two_body_point
 * and DetailedPatchySwapInteraction::_spherical_patchy_two_body in contrib/rovigatti
 * @param p
 * @param q
 * @param update_forces
 * @return
 */
number RaspberryInteraction::repulsive_pt_interaction(BaseParticle *p, BaseParticle *q, bool update_forces) {
    int p_type = m_ParticleList[p->get_index()];
    int q_type = m_ParticleList[q->get_index()];
    assert(p != q);

    // we use a Lennard Jones function with quadatic smoothin

    number e_lj;
    number energy = 0.;
    number rstar, b, rc, rsum;
    LR_vector force;
    LR_vector p_torque, q_torque;
    int ppidx, qqidx;
    number r_patch;
    // TODO: make this combinatorics problem less shit
    for (int pp = 0; pp < std::get<PTYPE_REP_PTS>(m_ParticleTypes[p_type]).size(); pp++){
        for (int qq = 0; qq < std::get<PTYPE_REP_PTS>(m_ParticleTypes[q_type]).size(); qq++) {
            // find interaction site positions for p & q
            LR_vector ppos = getParticleInteractionSitePosition(p, pp);
            LR_vector qpos = getParticleInteractionSitePosition(q, qq);
            // repulsive spheres have variable sizes,
            // we precompute the rmaxes-squareds for each pair of interaction types
            ppidx = std::get<PTYPE_REP_PTS>(m_ParticleTypes[p_type])[pp];
            qqidx = std::get<PTYPE_REP_PTS>(m_ParticleTypes[q_type])[qq];
            // lookup r-max squared
            // todo: probably possible to precompute these to save a little time
            // sum of radii of patchy particles
            rsum = get_r_sum(ppidx, qqidx);
            // todo: remove this when done testing
            if (rsum != std::get<REPULSION_DIST>(m_RepulsionPoints[ppidx]) + std::get<REPULSION_DIST>(m_RepulsionPoints[qqidx])) {
                throw oxDNAException("Mismatch between cached and computed rsum!");
            }

            // compute square of radial cutoff
            // for reasons i don't fully get, radial cutoff is *slightly* less than 1.0
//          number rmax_dist_sqr = get_r_max_sqr(ppidx, qqidx);

            // sigma
            number sigma_sqr = SQR(PLEXCL_S * rsum);

            // compute displacement vector between repulsion sites
            LR_vector rep_pts_dist = _computed_r + qpos - ppos;
            // distance-squared between the two repulsion pts
            number rep_pts_dist_sqr = rep_pts_dist.norm();
            // if the distance between the two interaction points is greater than the cutoff
            // we (josh speculating) set the cutoff JUST below 1 (where lj(1) = 0) to make calculations nicer
            number rmax_dist_sqr = SQR(rsum * PLEXCL_RC) ;

            if (rep_pts_dist_sqr < rmax_dist_sqr) {
                // unlike in other impls our model does not assume repulsive spheres have radius=0.5
                // r-factor = the sum of the radii of the repulsive interaction spheres, minus 1

                rc = PLEXCL_RC * rsum;

                // compute quadratic smoothing cutoff rstar
                rstar = PLEXCL_R * rsum;
                // if r is less than rstar, use the lennard-jones equation
                if(rep_pts_dist_sqr < SQR(rstar)) {
                    // partial lennard-jones value (sigma^2 / r^2)
                    number lj_partial = (sigma_sqr / rep_pts_dist_sqr);
                    // lj partial = (sigma/r)^6 = (sigma^2 / r^2) ^ 3
                    lj_partial = lj_partial * lj_partial * lj_partial;

                    // compute lennard-jones interaction energy
                    // (sigma / r) ^ 12 = ((sigma / r)^6)^2
                    e_lj = 4 * PLEXCL_EPS * (lj_partial * lj_partial - lj_partial);
                    // update energy
                    energy += e_lj;
                    if(update_forces) {
                        // compute forces
                        // i really hope this is correct
                        force = -rep_pts_dist * (24 * PLEXCL_EPS * (lj_partial - 2 * SQR(lj_partial)) / rep_pts_dist_sqr);
                    }
                }
                else {
                    // if r > rstar, use quadratic smoothing
                    // Vsmooth = b(xc - x)^2
                    // actual distance value, computed by taking the square root of rsquared
                    r_patch = sqrt(rep_pts_dist_sqr);
                    number rrc = r_patch - rc;

                    b = PLEXCL_B / SQR(rsum);

                    energy += PLEXCL_EPS * b * SQR(rrc);
                    if(update_forces) {
                        force = -rep_pts_dist * (2 * PLEXCL_EPS * b * rrc / r_patch);
                    }
                }

                if (update_forces){
                    // update particle force vectors
                    p->force -= force;
                    q->force += force;
                    // compute torques
                    // i grabbed this eqn from Lorenzo's code
                    p_torque = -p->orientationT * ppos.cross(force);
                    q_torque =  q->orientationT * qpos.cross(force);
                    // update particle torque vectors
                    p->torque += p_torque;
                    q->torque += q_torque;
                }
                // if the radius is greater than the cutoff, we can just ignore it
            }
        }
    }
    return energy;
}

number RaspberryInteraction::patchy_pt_interaction_angmod(BaseParticle *p, BaseParticle *q, bool update_forces) {
    /**
     * computes the forces and energies from the interactions of patches on two particles
     * I've mostly copied this code from alloassembly, which is itself a modified version of
     * oxDNA_torsion, which is a modified version of Flavio Romano's PatchyShapeInteraction code
     */
    int c = 0;
    LR_vector ppatch_a1, qpatch_a1, r_dist_dir;
    // LR_vector tmptorquep(0, 0, 0);
    // LR_vector tmptorqueq(0, 0, 0);
    const ParticleType& p_type = m_ParticleTypes[p->type];
    const ParticleType& q_type = m_ParticleTypes[q->type];
    number energy = 0.;
    number ta1, tb1, f1, fa1, fb1, cosa1, cosb1, rdist;

    number rnorm = _computed_r.norm();

    // iter patches on particle p
    for(int ppatch_idx = 0; ppatch_idx < std::get<PTYPE_PATCH_IDS>(p_type).size(); ppatch_idx++) {
        // important: keep pi (index of patch in particle type) distinct from ppatch_tid (patch type global id)
        // lookup patch type id
        int ppatch_tid = std::get<PTYPE_PATCH_IDS>(p_type)[ppatch_idx];

        LR_vector ppatch = getParticlePatchPosition(p, ppatch_idx);

        // iter patches on particle q
        for(int qpatch_idx = 0; qpatch_idx < std::get<PTYPE_PATCH_IDS>(q_type).size(); qpatch_idx++) {
            // note: do NOT use the position from this, since it's not rotated
            int qpatch_tid = std::get<PTYPE_PATCH_IDS>(q_type)[qpatch_idx];

            number e_2patch = 0.;

            // if patches can interact based on color, and both patches are active and not bound to another patch
            // todo maybe pass patch types so we don't need to recompute these in patches_can_interact
            if (patches_can_interact(p, q,
                                     ppatch_idx,
                                     qpatch_idx))
            {
                LR_vector qpatch = getParticlePatchPosition(q, qpatch_idx);
                // epsilon = interaction strength
                number eps = std::get<PP_INT_EPS>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);

                // get displacement vector between patches p and q
                // DO NOT apply particle orientation here! that is already applied!
                LR_vector patch_dist = _computed_r + qpatch - ppatch;
                // compute distancesquared
                number patch_dist_sqr = patch_dist.norm();
                //LR_vector<number> patch_dist_dir = patch_dist / sqrt(dist);
                //number rdist = sqrt(rnorm);
                //LR_vector<number> r_dist_dir = _computed_r / rdist;

                //printf("Patches %d and %d distance %f  cutoff is: %f,\n",pp->patches[pi].id,qq->patches[pj].id,dist,SQR(PATCHY_CUTOFF));

                number dist_sqr_cutoff = std::get<PP_MAX_DIST_SQR>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);
                if (patch_dist_sqr < dist_sqr_cutoff) {
                    // wait till now to look up either patch align, since most of the time we won't get this far

                    // compute ( r^2 )^4 / alpha
                    // alpha ** 10
                    number alpha_exp = std::get<PP_INT_ALPHA_POW>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);
                    // cocmpute r^8 / a^10 here so we can use it for forces later
                    number r8b10;
                    number exp_part = eps * compute_energy(patch_dist_sqr, alpha_exp, r8b10);

                    // keep exp_part seperate so it can be used in force calculation
                    number e_ij = exp_part;


                    // one-axis angle modulation
                    rdist = sqrt(rnorm);

                    r_dist_dir = _computed_r / rdist;

                    qpatch_a1 = getParticlePatchAlign(q, qpatch_idx);
                    ppatch_a1 = getParticlePatchAlign(p, ppatch_idx);
                    // cosine of angle between a1 and the patch displacement vector
                    cosa1 = ppatch_a1 * r_dist_dir;
                    // negative cosine of angle between a2 and the patch displacement vector
                    cosb1 = -qpatch_a1 * r_dist_dir;

                    ta1 = LRACOS(cosa1);
                    tb1 = LRACOS(cosb1);

                    fa1 =  narrow_type.V_mod(ta1);
                    fb1 =  narrow_type.V_mod(tb1) ;
                    f1 =  -eps * (exp_part - _patch_E_cut);
                    number angmod = f1 * fa1 * fb1;
                    assert(angmod >= 0 && angmod < 1.001);
                    e_ij *= angmod;


                    e_2patch += e_ij;

                    if (update_forces) {

                        number f1D = (5 * exp_part * r8b10);

                        LR_vector tmp_force = patch_dist * f1D;
                        LR_vector torquep, torqueq;
                        number fa1Dsin = narrow_type.V_modDsin(ta1);
                        number fb1Dsin = narrow_type.V_modDsin(tb1);

                        LR_vector dir;

                        //torque VM1
                        dir = r_dist_dir.cross(ppatch_a1);
                        torquep = dir * (f1 * fa1Dsin * fb1 );

                        //torque VM2
                        dir = r_dist_dir.cross(qpatch_a1);
                        torqueq = dir * (f1 * fa1 * fb1Dsin );


                        torquep += ppatch.cross(tmp_force);
                        torqueq += qpatch.cross(tmp_force);

                        // i'm not sure why this comes after the torque but that's how it is in romano
                        tmp_force += (ppatch_a1 - r_dist_dir * cosa1) * (f1 * fa1Dsin * fb1 / rdist);
                        tmp_force += -(qpatch_a1 + r_dist_dir * cosb1) * (f1 * fa1 * fb1Dsin / rdist);

                        p->force -= tmp_force;
                        q->force += tmp_force;

                        // TODO: better torque?
                        LR_vector p_torque = p->orientationT * torquep;
                        LR_vector q_torque = q->orientationT * torqueq;

                        p->torque -= p_torque;
                        q->torque += q_torque;
                    }

                }
            }
            // if energy is great large enough for bond
            if (e_2patch < getPatchBondEnergyCutoff()) {
                // if bond does not alread exist
                if (!is_bound_to(p->index, ppatch_idx, q->index, qpatch_idx)) {
                    // set bound
                    set_bound_to(p->index, ppatch_idx, q->index, qpatch_idx);
                }
            }
            // if energy is not small enough for a bond, but one exists
            else if (is_bound_to(p->index, ppatch_idx, q->index, qpatch_idx)){
                // if patches are bound, but shouldn't be
                clear_bound_to(p->index, ppatch_idx);
                clear_bound_to(q->index, qpatch_idx);

            }
            energy += e_2patch;
        }
    }
    return energy;
}

number RaspberryInteraction::patchy_pt_interaction_noangmod(BaseParticle *p, BaseParticle *q, bool update_forces) {
    /**
     * computes the forces and energies from the interactions of patches on two particles
     * I've mostly copied this code from alloassembly, which is itself a modified version of
     * oxDNA_torsion, which is a modified version of Flavio Romano's PatchyShapeInteraction code
     */
    int c = 0;
    const ParticleType& p_type = m_ParticleTypes[p->type];
    const ParticleType& q_type = m_ParticleTypes[q->type];
    number energy = 0.;

    // iter patches on particle p
    for(int ppatch_idx = 0; ppatch_idx < std::get<PTYPE_PATCH_IDS>(p_type).size(); ppatch_idx++) {
        // important: keep pi (index of patch in particle type) distinct from ppatch_tid (patch type global id)
        // lookup patch type id
        int ppatch_tid = std::get<PTYPE_PATCH_IDS>(p_type)[ppatch_idx];

        LR_vector ppatch = getParticlePatchPosition(p, ppatch_idx);

        // iter patches on particle q
        for(int qpatch_idx = 0; qpatch_idx < std::get<PTYPE_PATCH_IDS>(q_type).size(); qpatch_idx++) {
            // note: do NOT use the position from this, since it's not rotated
            int qpatch_tid = std::get<PTYPE_PATCH_IDS>(q_type)[qpatch_idx];

            number e_2patch = 0.;

            // if patches can interact based on color, and both patches are active and not bound to another patch
            // todo maybe pass patch types so we don't need to recompute these in patches_can_interact
            bool can_interact = patches_can_interact(p, q,
                                     ppatch_idx,
                                     qpatch_idx);
            if (can_interact)
            {
                LR_vector qpatch = getParticlePatchPosition(q, qpatch_idx);
                number eps = std::get<PP_INT_EPS>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);

                // get displacement vector between patches p and q
                // DO NOT apply particle orientation here! that is already applied!
                LR_vector patch_dist = _computed_r + qpatch - ppatch;
                // compute distancesquared
                number patch_dist_sqr = patch_dist.norm();
                //LR_vector<number> patch_dist_dir = patch_dist / sqrt(dist);
                //number rdist = sqrt(rnorm);
                //LR_vector<number> r_dist_dir = _computed_r / rdist;

                //printf("Patches %d and %d distance %f  cutoff is: %f,\n",pp->patches[pi].id,qq->patches[pj].id,dist,SQR(PATCHY_CUTOFF));

                number dist_sqr_cutoff = std::get<PP_MAX_DIST_SQR>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);
                if (patch_dist_sqr < dist_sqr_cutoff) {
                    // wait till now to look up either patch align, since most of the time we won't get this far

                    // compute ( r^2 )^4 / alpha
//                    const Patch& ppatchtype = getParticlePatchType(p, ppatch_idx);
//                    const Patch& qpatchtype = getParticlePatchType(q, qpatch_idx);
                    // alpha ** 10
                    number alpha_exp = std::get<PP_INT_ALPHA_POW>(m_PatchPatchInteractions[{ppatch_tid, qpatch_tid}]);
                    // cocmpute r^8 / a^10 here so we can use it for forces later
                    number r8b10;
                    number exp_part = eps * compute_energy(patch_dist_sqr, alpha_exp, r8b10);
                    e_2patch += exp_part;

                    if (update_forces) {
                        // lorenzo version
//                        // i gotta deal with this
//                        number force_mod = eps * a_part * exp_part * (4. * b_part / (SQR(r_p) * r_p)) +
//                                           sigma_ss * tmp_energy / SQR(r_p - rcut_ss);
//                        LR_vector tmp_force = patch_dist * (force_mod / r_p);
                        // compute force magnitude
                        number f1D = (5 * exp_part * r8b10);

                        LR_vector tmp_force = patch_dist * f1D;

                        p->force -= tmp_force;
                        q->force += tmp_force;

                        // TODO: better torque?
                        LR_vector p_torque = p->orientationT * ppatch.cross(tmp_force);
                        LR_vector q_torque = q->orientationT * qpatch.cross(tmp_force);

                        p->torque -= p_torque;
                        q->torque += q_torque;
                    }

                }
            }
            // if energy is great large enough for bond
            if (e_2patch < getPatchBondEnergyCutoff()) {
                // if bond does not alread exist
                if (!is_bound_to(p->index, ppatch_idx, q->index, qpatch_idx)) {
                    // set bound
                    set_bound_to(p->index, ppatch_idx, q->index, qpatch_idx);
                }
            }
            // if energy is not small enough for a bond, but one exists
            else if (is_bound_to(p->index, ppatch_idx, q->index, qpatch_idx)){
                // if patches are bound, but shouldn't be
                clear_bound_to(p->index, ppatch_idx);
                clear_bound_to(q->index, qpatch_idx);

            }
            energy += e_2patch;
        }
    }
    return energy;
}

number RaspberryInteraction::compute_energy(number patch_dist_sqr, number alpha_exp, number &r8b10) {
    // exponential factor (for debugging)
    r8b10 = patch_dist_sqr * patch_dist_sqr * patch_dist_sqr * patch_dist_sqr / alpha_exp;
    // exponential part = -1.001 e^((r/alpha)
    return  -1.001 * exp(-r8b10 * patch_dist_sqr);
}

/**
 *
 * @param p
 * @param q
 * @param ppatch_idx index of patch to check in p (NOT TYPE ID)
 * @param qpatch_idx  index of patch to check in q (NOT TYPE ID)
 * @return
 */
bool RaspberryInteraction::patches_can_interact(BaseParticle *p,
                                                BaseParticle *q,
                                                int ppatch_idx,
                                                int qpatch_idx) const {

    const Patch& ppatch_type = getParticlePatchType(p, ppatch_idx);
    const Patch& qpatch_type = getParticlePatchType(q, qpatch_idx);

    // if patch types can't interact, these two patches can't interact. full stop.
    if (!patch_types_interact(ppatch_type, qpatch_type)){
        return false;
    }
    // if either patch isn't active, return false
    if (!patch_is_active(p, ppatch_type)){
        return false;
    }
    if (!patch_is_active(q, qpatch_type)){
        return false;
    }
    if (patch_bound(p, ppatch_idx) && patch_bound_to(p, ppatch_idx) != std::pair<int,int>(q->index, qpatch_idx)){
        return false;
    }
    // todo: make this more efficient
    if (patch_bound(q, qpatch_idx) && patch_bound_to(q, qpatch_idx) != std::pair<int,int>(p->index, ppatch_idx)){
        return false;
    }
    // todo: find and report asymmetric locks
    return true;
}


bool RaspberryInteraction::patch_is_active(BaseParticle* p, const Patch& patch_type) const {
    int state_var = std::get<PPATCH_STATE>(patch_type);
    if (state_var == 0){
        return true; // var 0 = identity variable
    }
    // if state var < 0, var is virtual
    if (state_var < 0){
        return !m_ParticleStates[p->index][-state_var];
    }
    else {
        return m_ParticleStates[p->index][state_var];
    }
}

bool RaspberryInteraction::patch_types_interact(const RaspberryInteraction::Patch &ppatch_type,
                                                const RaspberryInteraction::Patch &qpatch_type) const {
    return m_PatchPatchInteractions.count(
            {
                std::get<PPATCH_TYPE_ID>(ppatch_type),
                std::get<PPATCH_TYPE_ID>(qpatch_type)
            }) > 0 &&
            std::get<PP_INT_EPS>(m_PatchPatchInteractions.at(
            {
                std::get<PPATCH_TYPE_ID>(ppatch_type),
                std::get<PPATCH_TYPE_ID>(qpatch_type)
            }
        )) != 0.;
}

/**
 * checks if patch is bound to anything
 * @param p
 * @param patch_idx
 * @return
 */
bool RaspberryInteraction::patch_bound(BaseParticle *p, int patch_idx) const {
    return patch_bound_to(p, patch_idx).first != -1;
}

/**
 * @param p
 * @param patch_idx index WITHIN THE PARTICLE TYPE, NOT PATCH TYPE ID
 * @return
 */
const RaspberryInteraction::ParticlePatch& RaspberryInteraction::patch_bound_to(BaseParticle *p,
                                                                                int patch_idx) const {
    assert(m_PatchyBonds[p->get_index()].size() > patch_idx);
    return m_PatchyBonds[p->get_index()][patch_idx];
}

bool RaspberryInteraction::is_bound_to(int p, int ppatch_idx, int q, int qpatch_idx) const {
    assert(p < m_PatchyBonds.size());
    assert(ppatch_idx < m_PatchyBonds[p].size());
    assert(q < m_PatchyBonds.size());
    assert(qpatch_idx < m_PatchyBonds[q].size());
    return m_PatchyBonds[p][ppatch_idx] == ParticlePatch(q, qpatch_idx);
}

void RaspberryInteraction::set_bound_to(int p, int ppatch_idx, int q, int qpatch_idx) {
    assert(p < m_PatchyBonds.size());
    assert(ppatch_idx < m_PatchyBonds[p].size());
    assert(p < m_PatchyBonds.size());
    assert(qpatch_idx < m_PatchyBonds[q].size());
    m_PatchyBonds[p][ppatch_idx] = {q,qpatch_idx};
    m_PatchyBonds[q][qpatch_idx] = {p,ppatch_idx};
}

void RaspberryInteraction::clear_bound_to(int p, int ppatch_idx) {
    assert(p < m_PatchyBonds.size());
    assert(ppatch_idx < m_PatchyBonds[p].size());
    m_PatchyBonds[p][ppatch_idx] = {-1,-1};
}

const std::vector<RaspberryInteraction::ParticlePatch> &RaspberryInteraction::getBondsFor(int idx) const {
    assert(idx < m_PatchyBonds.size());
    assert(idx > -1);
    return m_PatchyBonds[idx];
}

/**
 * returns the sum of the two radii of the interaction sites given
 * @param intSite1 type ID of first interaction point
 * @param intSite1 type ID of first interaction point
 * @return
 */
number RaspberryInteraction::get_r_sum(const int &intSite1, const int &intSite2) const {
    return m_RepulsionDistSums.at({intSite1, intSite2});
}

/**
 * retrieves the maximum distance at which two repulsive interaction point types (that's a mouthful!)
 * will interact
 * @param intSite1 type ID of first interaction point
 * @param intSite2 type ID of second interaction point
 * @return
 */
number RaspberryInteraction::get_r_max_sqr(const int &intSite1, const int &intSite2) const {
    return m_RepulsionDistSqrSumMaxs.at({intSite1, intSite2});
}

bool RaspberryInteraction::canChangeState(int particleIdx) const {
    return m_ParticlesCanChangeState.find(particleIdx) != m_ParticlesCanChangeState.end();
}

const std::vector<bool>& RaspberryInteraction::getParticleState(int particleIdx) const {
    assert(particleIdx < m_ParticleStates.size());
    assert(particleIdx > -1);
    return m_ParticleStates[particleIdx];
}

int stateValue(const std::vector<bool>& stateVec, int stateSize) {
    int value = 0;
    // note: we start at 1 since var 0 is the identity variable and doesn't actually represent a state
    for (int i = 1; i < stateSize; i++) {
        if (stateVec[i]) {
            value += (1 << i);
        }
    }
    return value;
}

std::string readLineNoComment(std::istream& inp){
    std::string sz;
    do {
        // read first line
        std::getline(inp, sz);
        sz = Utils::trim(sz);

    } while (sz.length() > 0 && sz[0] == '#'); // skip comments
    return sz;
}

LR_vector parseVector(const std::string& vec){
    number tmpf[3];
    int tmpi = sscanf(vec.c_str(), "%lf,%lf,%lf", tmpf, tmpf + 1, tmpf + 2);
    if(tmpi != 3)
        throw oxDNAException ("Could not parse vector %s. Aborting", vec.c_str());
    return {tmpf[0], tmpf[1], tmpf[2]};
}

extern "C" RaspberryInteraction* make_RaspberryPatchyInteraction() {
    return new RaspberryInteraction();
}