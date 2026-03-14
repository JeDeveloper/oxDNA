//
// Created by josh on 3/12/26.
//

#include "RaspberryPatchyBonds.h"
#include "../Interactions/RaspberryInteraction.h"

/**
 * encodes the bonds formed by the pathcy particles as a multidigraph
 * @param curr_step current simulation step
 * @return patchy bonds as strings
 */
std::string RaspberryPatchyBonds::get_output_string(llint curr_step) {
    RaspberryInteraction *interction = static_cast<RaspberryInteraction *>(_config_info->interaction);
    std::string sz;
    std::set<std::set<std::pair<int,int>>> unique_bonds; // avoid double counting
    for (int i = 0; i < _config_info->N(); i++) {
        const std::vector<RaspberryInteraction::ParticlePatch> &bonds = interction->getBondsFor(i);
        for (unsigned int j = 0; j < bonds.size(); j++) {
            int other_particle = bonds[j].first;
            int other_patch = bonds[j].second;
            if (other_particle != -1) {
                std::set<std::pair<int,int>> bond = {{i,j}, {other_particle, other_patch}};
                if (unique_bonds.find(bond) == unique_bonds.end()) {
                    unique_bonds.insert(bond);
                    sz += Utils::sformat("((%d %d), (%d, %d)) ", i, j, other_particle, other_patch);
                }
            }
        }
    }
    return sz;
}