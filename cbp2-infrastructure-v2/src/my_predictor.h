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
};

class my_predictor : public branch_predictor {
public:
	my_update u;
	branch_info bi;
	int dot_product;
	map<unsigned int, vector<int> > perceptrons;
	unsigned int history;

	my_predictor(void) : history(0) {

	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.addr = b.address;
			map<unsigned int, vector<int> >::iterator fetch = perceptrons.find(b.address);
			if (fetch != perceptrons.end()) {
				vector<int> p = fetch->second;
				dot_product = p[0];
				unsigned int mask = 2;
				for (int i = 1; i < WEIGHTS_LEN; i++) {
					dot_product += (history & mask) ? p[i] : -p[i];
					mask <<= 1;
				}
				u.direction_prediction (dot_product >= 0);
			} else {
				dot_product = 1;
				u.direction_prediction (true);
			}
		} else {
			u.direction_prediction (true);
		}
		u.target_prediction (1);
		return &u;
	}

	void update (branch_update *u, bool taken, unsigned int target) {
		if (bi.br_flags & BR_CONDITIONAL) {
			map<unsigned int, vector<int> >::iterator fetch = perceptrons.find(bi.address);
			if (fetch != perceptrons.end()) {
				vector<int> p = fetch->second;
				int t = taken ? 1 : -1;
				if (dot_product*t >= 0 || abs(dot_product) <= THETA) {
					p[0] += t;
					unsigned int mask = 2;
					for (int i = 1; i < WEIGHTS_LEN; i++) {
						p[i] += ((history & mask) >> i)==taken ? 1 : -1;
						mask <<= 1;
					}
				}
			} else {
				vector<int> p (HISTORY_LEN, taken);
				perceptrons.insert(pair<unsigned int, vector<int> > (bi.address, p));
			}
			history = (history << 1) | taken;
		}
	}
};
