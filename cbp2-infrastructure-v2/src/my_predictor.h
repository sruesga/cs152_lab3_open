#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <cmath>

using namespace std;

// my_predictor.h
// Attempt at building a perceptron-based branch predictor.
// Based on the following documentation:
// https://www.cs.utexas.edu/~lin/papers/hpca01.pdf

#define HISTORY_LEN 62
#define WEIGHTS_LEN HISTORY_LEN+1
#define THETA 1.93*HISTORY_LEN + 14

class my_update : public branch_update {
public:
	unsigned int addr;
	int dot_product;
};

class my_predictor : public branch_predictor {
public:
	my_update u;
	branch_info bi;
	map<unsigned int, vector<int> > perceptrons;
	unsigned int history;

	map<unsigned int, vector<int> >::iterator fetch;
	vector<int> p;

	my_predictor(void) : history(0) {

	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.addr = b.address;
			fetch = perceptrons.find(b.address);
			if (fetch != perceptrons.end()) {
				p = fetch->second;
				u.dot_product = p[0];
				unsigned int mask = 1 << (HISTORY_LEN - 1);
				for (int i = 1; i < WEIGHTS_LEN; i++) {
					u.dot_product += (history & mask) ? p[i] : -p[i];
					mask >>= 1;
				}
				u.direction_prediction (u.dot_product >= 0);
			} else {
				u.dot_product = 1;
				u.direction_prediction (true);
			}
		} else {
			u.direction_prediction (true);
		}
		u.target_prediction (1);
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		my_update* w = (my_update*) u;
		my_update v = *w;
		if (bi.br_flags & BR_CONDITIONAL) {
			fetch = perceptrons.find(v.addr);
			if (fetch != perceptrons.end()) {
				p = fetch->second;
				int t = taken ? 1 : -1;
				if (v.dot_product*t >= 0 || abs(v.dot_product) <= THETA) {
					p[0] += t;
				unsigned int mask = 1 << (HISTORY_LEN - 1);
					for (int i = 1; i < WEIGHTS_LEN; i++) {
						p[i] += ((history & mask) >> i)==taken ? 1 : -1;
						mask >>= 1;
					}
				}
			} else {
				perceptrons.insert(pair<unsigned int, vector<int> > (v.addr, vector<int> (HISTORY_LEN, taken)));
			}
			history = (history << 1) | taken;
		}
	}
};
