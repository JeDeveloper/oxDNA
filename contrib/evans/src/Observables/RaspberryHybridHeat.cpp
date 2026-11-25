#include "RaspberryHybridHeat.h"
#include "../Interactions/RaspberryInteraction.h"
#include "Utilities/Utils.h"

std::string RaspberryHybridHeat::get_output_string(llint curr_step) {
    RaspberryInteraction *interaction = static_cast<RaspberryInteraction *>(_config_info->interaction);
    return Utils::sformat("%.10g", interaction->getHybridHeat());
}
