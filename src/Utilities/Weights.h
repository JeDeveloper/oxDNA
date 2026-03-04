/**
 * @file    Weights.h
 * @date    Feb 10, 2012
 * @author  flavio
 * 
 */

#ifndef WEIGHTS_H_
#define WEIGHTS_H_

#include "OrderParameters.h"

// if you really need to you can go above this
#define WEIGHT_MAT_MAX_SIZE 1e8

/// Weight class
class Weights {
protected:
	// weight matrix, linearized
	std::vector<double> _w;
	// total size of the linearized weight matrix
	int _dim;
	// number of dimensions of the weight matrix
	int _ndim;
	std::vector<int> _sizes;
public:
	Weights();
	~Weights();
	int get_dim () const { return _dim; }
	double get_weight_by_index (int) const;
	double get_weight(const vector<int> &state) const;
	double get_weight(const vector<int> &, int *) const;
	double get_weight(OrderParameters &) const;
	double get_weight(OrderParameters &, int *) const;
	void init(const char *, OrderParameters *, bool safe, double default_weight);
	void print();
};

#endif /* WEIGHTS_H_ */

