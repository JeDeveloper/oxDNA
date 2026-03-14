//
// Created by josh on 3/12/26.
//

#ifndef OXDNA_RASPBERRYPATCHYBONDS_H
#define OXDNA_RASPBERRYPATCHYBONDS_H
#include "../Observables/BaseObservable.h"


class RaspberryPatchyBonds : public BaseObservable {
public:
    std::string get_output_string(llint curr_step) override;

};

extern "C" BaseObservable *make_RaspberryPatchyBonds() {
    return new RaspberryPatchyBonds();
}


#endif //OXDNA_RASPBERRYPATCHYBONDS_H