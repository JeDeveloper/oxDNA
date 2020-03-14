/*
 * BaseList.cpp
 *
 *  Created on: 14 ott 2019
 *      Author: lorenzo
 */

#include "BaseList.h"

void BaseList::get_settings(input_file &inp) {
	char sim_type[512] = "MD";
	getInputString(&inp, "sim_type", sim_type, 0);

	// this chain of ifs is here so that if new sim_types are implemented
	// lists have to know how to manage the associated neighbouring lists
	if(strncmp("MD", sim_type, 512) == 0) _is_MC = false;
	else if(strncmp("MC", sim_type, 512) == 0) _is_MC = true;
	else if(strncmp("MC2", sim_type, 512) == 0) _is_MC = true;
	else if(strncmp("VMMC", sim_type, 512) == 0) _is_MC = true;
		else if(strncmp("PT_VMMC", sim_type, 512) == 0) _is_MC = true;
	else if(strncmp("FFS_MD", sim_type, 512) == 0) _is_MC = false;
	else if(strncmp("min", sim_type, 512) == 0) _is_MC = false;
	else if(strncmp("FIRE", sim_type, 512) == 0) _is_MC = false;
	else throw oxDNAException("BaseList does not know how to handle a '%s' sim_type\n", sim_type);
}

std::vector<BaseParticle *> BaseList::get_all_neighbours(BaseParticle *p) {
//	std::vector<BaseParticle *> neighs = std::move(get_complete_neigh_list(p));
	std::vector<BaseParticle *> neighs = get_complete_neigh_list(p);

	std::set<BaseParticle *> bonded_neighs;
	typename std::vector<ParticlePair >::iterator it = p->affected.begin();
	for(; it != p->affected.end(); it++) {
		if(it->first != p) bonded_neighs.insert(it->first);
		if(it->second != p) bonded_neighs.insert(it->second);
	}

	neighs.insert(neighs.end(), bonded_neighs.begin(), bonded_neighs.end());
	return neighs;
}

std::vector<ParticlePair > BaseList::get_potential_interactions() {
	std::vector<ParticlePair > list;

	for(uint i = 0; i < _particles.size(); i++) {
		BaseParticle *p = _particles[i];
//		std::vector<BaseParticle *> neighs = std::move(get_all_neighbours(p));
		std::vector<BaseParticle *> neighs = get_all_neighbours(p);
		typename std::vector<BaseParticle *>::iterator it = neighs.begin();
		for(; it != neighs.end(); it++) {
			BaseParticle *q = *it;
			if(p->index > q->index) list.push_back(ParticlePair(p, q));
		}
	}

	return list;
}