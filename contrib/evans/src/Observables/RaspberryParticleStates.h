//
// Created by josh on 3/17/26.
//

#ifndef OXDNA_RASPBERRYPARTICLESTATES_H
#define OXDNA_RASPBERRYPARTICLESTATES_H
#include <Observables/BaseObservable.h>

#define READABILITY_LVL_UNREADABLE 0
#define READABILITY_LVL_READABLE 1

class RaspberryParticleStates : public BaseObservable {
public:
    void get_settings(input_file &my_inp, input_file &sim_inp) override;
    std::string get_output_string(llint curr_step) override;
protected:
    int _readabilty_lvl; // options 0 for very unreadable, 1 for somewhat readable, 2 for very readable (but also very long)
};

extern "C" BaseObservable *make_RaspberryParticleStates() {
    return new RaspberryParticleStates();
}

#endif //OXDNA_RASPBERRYPARTICLESTATES_H