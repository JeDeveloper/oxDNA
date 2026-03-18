//
// Created by josh on 3/17/26.
//
#include "RaspberryParticleStates.h"

#include "../Particles/RaspberryParticle.h"
#include "../Interactions/RaspberryInteraction.h"

/**
 * encodes the states of the raspberry particles as strings
 * @param curr_step current simulation step
 * @return particle states as strings
 */

std::string RaspberryParticleStates::get_output_string(llint curr_step) {
    std::string sz;
    auto* interaction = static_cast<RaspberryInteraction *>(CONFIG_INFO->interaction);

    if (_readabilty_lvl == READABILITY_LVL_UNREADABLE) {
        for (int i = 0; i < CONFIG_INFO->N(); i++) {
            const std::vector<bool> &state = interaction->getParticleState(i);
            sz += Utils::sformat("%d ", stateValue(state, state.size()));
        }
    }
    else if (_readabilty_lvl == READABILITY_LVL_READABLE) {
        for (int i = 0; i < CONFIG_INFO->N(); i++) {
            const std::vector<bool> &states = interaction->getParticleState(i);
            sz += Utils::sformat("(%d: ", i);
            for (unsigned int j = 0; j < states.size(); j++) {
                sz += states[j] ? "1" : "0";
            }
            sz += ") ";
        }
    }
    else {
        throw oxDNAException("Invalid readability level %d for RaspberryParticleStates observable!", _readabilty_lvl);
    }
    return sz;

}