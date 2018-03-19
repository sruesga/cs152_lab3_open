#include <map>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>
#include <utility>
#include <array>
#include <vector>
#include <algorithm>

// my_predictor.h
// Attempt at building a hybrid perceptron-based branch predictor w/ gshare.
// Based on the following documentation:
// https://www.cs.utexas.edu/~lin/papers/hpca01.pdf

class my_update : public branch_update {
public:
	unsigned int index;
	unsigned int branch_index;
	bool is_g;
};

class my_predictor : public branch_predictor {
public:
	// g-share components
	#define HISTORY_LENGTH	24
	#define TABLE_BITS	24
	unsigned int g_history;
	unsigned char tab[1<<TABLE_BITS];

	// saturating counter table, to choose between g-share and perceptrons
	std::map<unsigned int, char> selector;

	#define HISTORY_GROUPS 2
	#define HISTORY_SIZE 64
	int HISTORY_TOTAL;
	unsigned long long int TOP;
	unsigned long long int BOTTOM;

	std::map<unsigned int, std::vector<long long int> > perceptrons;
	std::map<unsigned int, long long int> dot_history;
	unsigned long long int history[HISTORY_GROUPS]; // smaller indexed int represents older history
	my_update u;
	branch_info bi;
	unsigned int history_counter;

	//stack variables for use only in the prediction method
	std::map<unsigned int, std::vector<long long int> >::iterator fetch;
	std::map<unsigned int, char>::iterator hybrid;
	int dot_product;

	//stack variables for use only in the update method
	int t;
	int carry;
	int next;

	my_predictor (void) : g_history(0){
		memset (tab, 0, sizeof (tab));
		HISTORY_TOTAL = HISTORY_SIZE * HISTORY_GROUPS;
		TOP = std::pow(2, HISTORY_SIZE - 1);
		BOTTOM = TOP - 1;
		history[0] = TOP; //x_0 set to 1 for bias
		carry = 0;
		history_counter = 0;
	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) { // neural net
			u.branch_index = b.address;
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
			hybrid = selector.find(b.address);
			if (hybrid != selector.end()) { // hybrid selector
				u.is_g = (hybrid->second > 3);
			} else {
				u.is_g = false;
			}
			if (!u.is_g) { // gshare
				u.index =
				  (g_history << (TABLE_BITS - HISTORY_LENGTH))
				^ (b.address & ((1<<TABLE_BITS)-1));
				u.direction_prediction (tab[u.index] >> 1);
			}
		} else {
			u.direction_prediction (true);
		}
		u.target_prediction (0);
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		if (bi.br_flags & BR_CONDITIONAL) {
			// gshare
			unsigned char *c = &tab[((my_update*)u)->index];
			if (taken) {
				if (*c < 3) (*c)++;
			} else {
				if (*c > 0) (*c)--;
			}
			g_history <<= 1;
			g_history |= taken;
			g_history &= (1<<HISTORY_LENGTH)-1;

			// neural net
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
			if (((my_update*)u)->direction_prediction() != taken || std::abs (dot_history[((my_update*)u)->branch_index]) <= int(1.93 * HISTORY_TOTAL + 14)) {
				fetch = perceptrons.find(((my_update*)u)->branch_index);
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
					perceptrons.insert( std::pair<unsigned int, std::vector<long long int> >(((my_update*)u)->branch_index, weights) );
				}
			}

			// hybrid update
			hybrid = selector.find(((my_update*)u)->branch_index);
			if (hybrid == selector.end()) {
				selector.insert( std::pair<unsigned int, char> (((my_update*)u)->branch_index, 0));
			}
			if (((my_update*)u)->direction_prediction() != taken) {
				selector[((my_update*)u)->branch_index] = std::min(7, selector[((my_update*)u)->branch_index] + 1);
			} else {
				selector[((my_update*)u)->branch_index] = std::max(0, selector[((my_update*)u)->branch_index] - 1);
			}
		}
	}
};
