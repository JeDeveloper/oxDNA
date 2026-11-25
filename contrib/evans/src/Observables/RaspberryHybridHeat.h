//
// Energy exchanged with the h bath in RaspberryInteraction's hybrid lock mode (cumulative, simulation units): lock
// formation, h moves and dissolution. Without a thermostat, total energy minus this is conserved up to integration
// error. Prints 0 in the classic mode.
//

#ifndef OXDNA_RASPBERRYHYBRIDHEAT_H
#define OXDNA_RASPBERRYHYBRIDHEAT_H
#include "../Observables/BaseObservable.h"

class RaspberryHybridHeat : public BaseObservable {
public:
    std::string get_output_string(llint curr_step) override;
};

extern "C" BaseObservable *make_RaspberryHybridHeat() {
    return new RaspberryHybridHeat();
}

#endif //OXDNA_RASPBERRYHYBRIDHEAT_H
