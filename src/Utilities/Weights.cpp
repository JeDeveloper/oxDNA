/*
 * OrderParameters.cpp
 *
 *  Created on: Feb 10, 2012
 *      Author: sulc
 */

#include <cfloat>
#include <sstream>
#include "Weights.h"

// functions for Weights class
// Constructor
Weights::Weights () {
	_dim = -1;
	_ndim = -1;
}

Weights::~Weights () = default;

/**
 * loads weights from a file
 * @param filename filename from which to load weights
 * @param op order parameters object for which weights will be loaded
 * @param safe if True, check that the number of weights in the file match the number of possible states defined by the order parameter.
 * @param default_weight default weight to use for states not explicitly assigned weights, if param `safe` is set to false
 */
void Weights::init (const char * filename, OrderParameters * op, bool safe, double default_weight) {
	ifstream inp;
	inp.open(filename, ios::in);

	OX_LOG(Logger::LOG_INFO, "(Weights.cpp) Weight parser: parsing `%s\'", filename);

	if (!inp.good()) throw oxDNAException ("(Weights.cpp) parser: Can't read file `%s\'. Aborting", filename);

	// initialise the array
	_ndim = op->get_all_parameters_count(); // count order parameters
	_sizes.assign(op->get_state_sizes().data(), op->get_state_sizes().data() + _ndim); // list order parameter lengths

    _dim = 1;
	// calculate the dimension of linear array by multiplying together the sizes of each order parameter
    for (int i = 0; i < _ndim; i ++) _dim *= _sizes[i];

	if (_dim > WEIGHT_MAT_MAX_SIZE) {
		throw oxDNAException("Requesting to allocate a weight matrix of size %d, which is too big. You can increase the max weight matrix size by editing Weights.h if you really want to.", _dim);
	}

    _w.resize(_dim);
    for (int i = 0; i < _dim; i ++)  _w[i] = default_weight;

    OX_LOG (Logger::LOG_INFO, "(Weights.cpp) weights found; O.P. dim: %d, tot size: %d", _ndim, _dim);

    std::string line;
    int lineno = 1;
    while (inp.good()) {
        getline (inp, line);

        if (line.empty()) continue;

        // Strip leading whitespace
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#') continue;

        // Parse using istringstream instead of manual sscanf/pointer walking
        std::istringstream iss(line);
        std::vector<int> tmp(_ndim);
        double tmpf;
        int check = 0;

        for (int i = 0; i < _ndim; i++) {
            if (iss >> tmp[i]) check++;
        }
        if (iss >> tmpf) check++;

        if (check != _ndim + 1) throw oxDNAException ("(Weights.cpp) weight parser: error parsing line %d in %s. Not enough numbers in line. Aborting", lineno, filename);

        // Check that we're within the order parameter boundries
        for (int i = 0; i < _ndim; i ++) {
            if (tmp[i] < 0 || tmp[i] > (_sizes[i] + 2)) {
                throw oxDNAException ("(Weights.cpp) parser: error parsing line %d of `%s`': index %d out of OrderParameters bounds. Aborting\n", lineno, filename, tmp[i]);
            }
        }
    	if (tmpf < -DBL_EPSILON) {
    		throw oxDNAException ("(Weights.cpp) parser: error parsing line %d of `%s`': weight %lf < 0. Cowardly refusing to proceed. Aborting\n", lineno, filename, tmpf);
    	}

        int index = 0;
        for (int i = 0; i < _ndim; i ++) {
            int pindex = 1;
            for (int k = 0; k < i; k ++) {
                pindex *= _sizes[k];
            }
            index += tmp[i] * pindex;
        }

        if (index < _dim) _w[index] = tmpf;
        else OX_LOG (Logger::LOG_WARNING, "(Weights.cpp) Trying to assign weight to non-existent index %d. Weight file too long/inconsistent?", index);

        lineno ++;
    }

    lineno --;
    if (lineno != _dim) {
        if (safe) throw oxDNAException ("(Weights.cpp) number of lines in the weights file do not match the dimensions of the order parameter. Expecting %d lines, got %d.\n\tUse safe_weights = False and default_weight = <float> to assume a default weight. Aborting", _dim, lineno);
        else OX_LOG (Logger::LOG_INFO, "(Weights.cpp) number of lines in the weights file do not match the dimensions of the order parameter. Expecting %d lines, got %d. Using default weight %g", _dim, lineno, default_weight);
    }

    OX_LOG (Logger::LOG_INFO, "(Weights.cpp) parser: parsing done");
}

void Weights::print() {
	printf ("######## weights ##################\n");
	std::vector<int> tmp(_ndim);
	for (int i = 0; i < _dim; i ++) {
		for (int j = 0; j < _ndim; j ++) {
			int pindex = 1;
			for (int k = 0; k < j; k ++) {
				pindex *= _sizes[k];
			}
			tmp[j] = (i / pindex) % _sizes[j];
		}
		printf ("%d %lf %lf\n", i, _w[i], get_weight(tmp));
	}
	printf ("###################################\n");
}

/**
 *
 * @param index position in linearized weight matrix
 * @return value of linearized weight matrix at index
 */
double Weights::get_weight_by_index (const int index) const {
	return _w[index];
}

/**
 * get weight for state and assigns value of *ptr to the index in the linearized weight matrix(?)
 * @param state state
 * @param ptr pointer to memory to recieve index in linearized weight matrix
 * @return weight at state
 */
double Weights::get_weight(const vector<int> &state, int * ptr) const {
	int index = 0;
	for (int i = 0; i < _ndim; i ++) {
		assert (state[i] < _sizes[i]);
		int pindex = 1;
		for (int k = 0; k < i; k ++) {
			pindex *= _sizes[k];
		}
		index += state[i] * pindex;
	}
	if (index >= _dim) {
		printf ("index > dim: %i > %i\n", index, _dim);
		for (int k = 0; k<_ndim; k ++) {
			printf ("%d ",state[k]);
		}
		for (int k = 0; k<_ndim; k ++) {
			printf ("%d ",state[k]);
		}
		printf ("\n");
	}
	* ptr = index;
	if (index >= _dim) {
		// build a readable string representation of `state`
		std::ostringstream ss;
		ss << "(";
		for (int k = 0; k < _ndim; ++k) {
			if (k) ss << ",";
			ss << state[k];
		}
		ss << ")";

		throw oxDNAException("Flattened value of state %s (%d) is outside the bounds of linearized weight matrix (length %d).",
			ss.str().c_str() ,index, _dim);
	}
	return _w[index];
}

/**
 * gets the weight associated with a system state
 * @param state state, represented as a series of values for order parameters
 * @return weight associated with state
 */
double Weights::get_weight(const vector<int> &state) const {
	int index = 0;
	for (int i = 0; i < _ndim; i ++) {
		assert (state[i] < _sizes[i]);
		int pindex = 1;
		for (int k = 0; k < i; k ++) {
			pindex *= _sizes[k];
		}
		index += state[i] * pindex;
	}
	if (index >= _dim) {
		printf ("index > dim: %i > %i\n", index, _dim);
		for (int k = 0; k<_ndim; k ++) {
			printf ("%d ",state[k]);
		}
		for (int k = 0; k<_ndim; k ++) {
			printf ("%d ",state[k]);
		}
		printf ("\n");
	}

	return _w[index];
}

/**
 * wrapper for get_weight(std::vector<int>&)
 * @param state
 * @return
 */
double Weights::get_weight(OrderParameters &state) const {
	//return get_weight (arg->get_hb_states());
	return get_weight (state.get_all_states());
}

/**
 * wrapper for get_weight(const std::vector<int>&, int* ptr)
 * @param state
 * @param ptr
 * @return
 */
double Weights::get_weight(OrderParameters &state, int * ptr) const {
	//return get_weight (arg->get_hb_states(), ptr);
	return get_weight (state.get_all_states(), ptr);
}
