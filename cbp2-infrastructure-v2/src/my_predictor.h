#include <list>
#include <map>
#include <iostream>
#include <string>
#include <cmath>

// my_predictor.h
// Attempt at building a perceptron-based branch predictor.
// Based on the following documentation:
// https://www.cs.utexas.edu/~lin/papers/hpca01.pdf

#define HISTORY_LENGTH 62
#define N 32
#define THETA 1.93*HISTORY_LENGTH + 14

class my_update : public branch_update {
public:
	unsigned int addr;
	int dot_product;
};

struct Perceptron
{
	int weights[N + 1]
}

class my_predictor : public branch_predictor {
public:
	my_update u;
	branch_info bi;
	std::map<unsigned int, Perceptron > perceptrons;
	int[HISTORY_LENGTH] history;

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.addr = b.address;
			p = perceptrons.fetch
			if (fetch != perceptrons.end()) {
				p = fetch->second;
				dot_product = 0;
                for (p_iter = p.begin(), h_iter = history.begin(); p_iter != p.end() && h_iter != history.end(); ++p_iter, ++h_iter) {
                    dot_product += (*p_iter * *h_iter);
                }
				u.direction_prediction (dot_product >= 0);
			} else {
				u.direction_prediction (true);
			}

		} else {
			u.direction_prediction (true);
		}
		u.target_prediction (0);
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		if (bi.br_flags & BR_CONDITIONAL) {
		}
	}
};
