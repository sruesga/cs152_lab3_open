#include <map>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <utility>
#include <array>

// my_predictor.h
// Attempt at building a perceptron-based branch predictor.
// Based on the following documentation:
// https://www.cs.utexas.edu/~lin/papers/hpca01.pdf

class my_update : public branch_update {
public:
	unsigned int index;
};

class my_predictor : public branch_predictor {
public:
	#define HISTORY_GROUPS 2
	#define HISTORY_SIZE 64
	int HISTORY_TOTAL;
	unsigned long long int TOP;
	unsigned long long int BOTTOM;

	std::map<unsigned int, std::array<int, HISTORY_SIZE> > perceptrons;
	std::map<unsigned int, int> dot_history;
	unsigned long long int history[HISTORY_GROUPS]; // smaller indexed int represents older history
	my_update u;
	branch_info bi;
	unsigned int history_counter;

	//stack variables for use only in the prediction method
	std::map<unsigned int, std::array<int, HISTORY_SIZE> >::iterator fetch;
	int dot_product;

	//stack variables for use only in the update method
	int t;
	int carry;
	int next;

	my_predictor (void) {
		HISTORY_TOTAL = HISTORY_SIZE * HISTORY_GROUPS;
		TOP = std::pow(2, HISTORY_SIZE - 1);
		BOTTOM = TOP - 1;
		history[0] = TOP; //x_0 set to 1 for bias
		carry = 0;
		history_counter = 0;
	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.index = b.address;
			fetch = perceptrons.find(b.address);
			if (fetch != perceptrons.end()) {
				dot_product = 0;
                for (int i = 0; i < history_counter; i += 1) {
					int result = (history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1;
					dot_product += (result == 1)? fetch->second[i] : -fetch->second[i];
                }
				dot_history[b.address] = dot_product;
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
			t = taken? 1 : 0;
			for (int i = HISTORY_GROUPS - 1; i >= 0; i -= 1){
				next = history[i] & TOP >> (HISTORY_SIZE - 1) & 1;
				history[i] <<= 1;
				history[i] |= carry;
				carry = next;
			}
			history[0] |= TOP;
			history[HISTORY_GROUPS - 1] |= t;
			history_counter += (history_counter < HISTORY_TOTAL)? 1 : 0;
			if (((my_update*)u)->direction_prediction() != taken || std::abs (dot_history[((my_update*)u)->index]) <= INT_MAX) {
				fetch = perceptrons.find(((my_update*)u)->index);
				if (fetch != perceptrons.end()) { // perceptron found, learn weights
                    for (int i = 0; i < HISTORY_TOTAL; i += 1) {
						fetch->second[i] += (taken? 1 : -1) * ((((history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1) == 1)? 1 : -1);
                    }
				} else { // perceptron yet to exist, add a new vector
					std::array<int, HISTORY_SIZE> weights;
					for (int i = 0; i < HISTORY_TOTAL; i += 1) {
						weights[i] = (taken? 1 : -1) * ((((history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1) == 1)? 1 : -1);
					}
					perceptrons.insert( std::pair<unsigned int, std::array<int, HISTORY_SIZE> >(((my_update*)u)->index, weights) );
				}
			}
		}
	}
};
