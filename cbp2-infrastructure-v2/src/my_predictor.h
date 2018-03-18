#include <list>
#include <map>
#include <iostream>
#include <string>

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
	std::map<unsigned int, std::list<int> > perceptrons;
	std::list<int> history;
	my_update u;
	branch_info bi;
    std::list<int>::iterator h_iter;

	//stack variables for use only in the prediction method
	std::map<unsigned int, std::list<int> >::iterator fetch;
    std::list<int>::iterator p_iter;
	std::list<int> p;
	int dot_product;

	//stack variables for use only in the update method
	int t;

	my_predictor (void) {
		history.push_back(1); //x_0 set to 1 for bias
	}

	branch_update *predict (branch_info & b) {
		bi = b;
		if (b.br_flags & BR_CONDITIONAL) {
			u.index = b.address;
			fetch = perceptrons.find(b.address);
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
			t = taken? 1 : -1;
			history.push_back (t);
			if (((my_update*)u)->direction_prediction() != taken || dot_product <= 0) {
				fetch = perceptrons.find(((my_update*)u)->index);
				if (fetch != perceptrons.end()) { // perceptron found, learn weights
					std::list<int> weights;
                    for (p_iter = fetch->second.begin(), h_iter = history.begin(); p_iter != fetch->second.end() && h_iter != history.end(); ++p_iter, ++h_iter) {
                        weights.push_back(*p_iter + (t * *h_iter));
                    }
                    for (;h_iter != history.end(); ++h_iter) {
                        weights.push_back(t * *h_iter);
                    }
					perceptrons[((my_update*)u)->index] = weights;
				} else { // perceptron yet to exist, add a new vector
					std::list<int> weights;
                    for (h_iter = history.begin(); h_iter != history.end(); ++h_iter) {
                        weights.push_back(t * *h_iter);
                    }
					perceptrons[((my_update*)u)->index] = weights;
				}
			}
		}
	}
};
