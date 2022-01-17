#ifndef NNETS_H
#define NNETS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>

using namespace std;

class nnets
{
public:
	nnets(ifstream& inFile);
	void train(ifstream& trainfile, int epoch, double learningRate);
	void test(ifstream& testFile, ostream& outfile);
	void print(ostream& outFile);

private:
	double sigmoid(double);
	double sigmoidPrime(double);
	struct edge;

	struct node {
		double activation;
		double input;
		double error;
		vector<edge> outEdges;
		vector<edge> inEdges;
	};
	struct edge {
		double weight;
		node* toNode;
	};

	vector<node> inputLayer;
	vector<node> hiddenLayer;
	vector<node> outputLayer;
	int numInput, numHidden, numOutput;
};

#endif
