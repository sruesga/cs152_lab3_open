#include <map>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <utility>
#include <array>
#include <vector>

// my_predictor.h
// Attempt at building an ideal TAGE predictor
// Based on the following documentation:
// http://www.irisa.fr/caps/people/seznec/JILP-COTTAGE.pdf

class my_update : public branch_update {
public:
	unsigned int index;
	unsigned int table_index;
};

class my_predictor : public branch_predictor {
public:

	#define HISTORY_GROUPS 2
	#define HISTORY_SIZE 64
	#define TABLE_GROUPS 8

	// variables for global history register
	int HISTORY_TOTAL;
	unsigned long long int TOP;
	unsigned long long int BOTTOM;

	// vector holds all maps, first entry is a standard 2-bit saturating counter
	std::vector< std::map< std::pair<unsigned int, std::vector<unsigned long long int> >, unsigned char > > tables;
	// vector representation of global history register
	std::vector<unsigned long long int> history;
	unsigned long long int history[HISTORY_GROUPS];
	my_update u;
	branch_info bi;
	unsigned int history_counter;

	//stack variables for use only in the prediction method
	std::vector< std::map< std::pair<unsigned int, std::vector<unsigned long long int> >, unsigned char > >::iterator fetch;
	int dot_product;

	//stack variables for use only in the update method
	int t;
	int carry;
	int next;

	my_predictor (void) {
		history.resize(HISTORY_GROUPS, 0);
		tables.resize(TABLE_GROUPS);
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
			for (int i = TABLE_GROUPS - 1; i >= 0; i -= 1) {

				fetch = tables[i].find (std::pair<unsigned int, std::vector<unsigned long long int> >(b.address, history) );
				if (fetch != tables[i].end()) {
					u.table_index = i;

				}
			}




			fetch = perceptrons.find(b.address);
			if (fetch != perceptrons.end()) {
				dot_product = fetch->second[0];
                for (int i = 1; i < history_counter; i += 1) {
					long long int result = (history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1;
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
			if (((my_update*)u)->direction_prediction() != taken || std::abs (dot_history[((my_update*)u)->index]) <= LLONG_MAX) {
				fetch = perceptrons.find(((my_update*)u)->index);
				if (fetch != perceptrons.end()) { // perceptron found, learn weights
                    for (int i = 0; i < HISTORY_TOTAL; i += 1) {
						fetch->second[i] += (taken? 1 : -1) * ((((history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1) == 1)? 1 : -1);
                    }
				} else { // perceptron yet to exist, add a new vector
					std::vector<long long int> weights;
					weights.resize(HISTORY_TOTAL);
					for (int i = 0; i < HISTORY_TOTAL; i += 1) {
						weights[i] = (taken? 1 : -1) * ((((history[(int) i / HISTORY_SIZE]) >> (HISTORY_SIZE - 1 - (i % HISTORY_SIZE)) & 1) == 1)? 1 : -1);
					}
					perceptrons.insert( std::pair<unsigned int, std::vector<long long int> >(((my_update*)u)->index, weights) );
				}
			}
		}
	}
};
