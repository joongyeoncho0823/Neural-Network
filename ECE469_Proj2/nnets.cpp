#include "nnets.h"

using namespace std;

nnets::nnets(ifstream& inFile) {
	inFile >> numInput >> numHidden >> numOutput; //30 5 1
	// one more node for bias
	inputLayer.resize(numInput + 1);
	hiddenLayer.resize(numHidden + 1);
	outputLayer.resize(numOutput + 1);

	inputLayer[0].activation = -1;
	hiddenLayer[0].activation = -1;
	outputLayer[0].activation = -1;

	for (int i = 1; i <= numHidden; i++) { // 1~5 (total 5)
		for (int j = 0; j <= numInput; j++) { // 0 ~ 30 (total 31)
			edge outEdge, inEdge;
			double weight;
			inFile >> weight;
			outEdge.weight = weight;
			outEdge.toNode = &hiddenLayer[i];
			inputLayer[j].outEdges.push_back(outEdge);

			inEdge.weight = weight;
			inEdge.toNode = &inputLayer[j];
			hiddenLayer[i].inEdges.push_back(inEdge);
		}
	}
	for (int i = 1; i <= numOutput; i++) {
		for (int j = 0; j <= numHidden; j++) {
			edge outEdge, inEdge;
			double weight;
			inFile >> weight;
			outEdge.weight = weight;
			outEdge.toNode = &outputLayer[i];
			hiddenLayer[j].outEdges.push_back(outEdge);

			inEdge.weight = weight;
			inEdge.toNode = &hiddenLayer[j];
			outputLayer[i].inEdges.push_back(inEdge);
		}
	}

	inFile.close();
}

void nnets::train(ifstream& trainfile, int epoch, double learningRate) {
	int numTrainingExamples, numIns, numOuts;
	trainfile >> numTrainingExamples >> numIns >> numOuts;  // 300 30 1

	struct trainingExample {
		vector<double> inputs;
		vector<double> output;
	};
	vector<trainingExample> examples;
	examples.resize(numTrainingExamples);

	for (int i = 0; i < numTrainingExamples; i++) {
		trainingExample tmp;
		tmp.inputs.resize(numIns);
		tmp.output.resize(numOuts);
		for (int j = 0; j < numIns; j++) {
			double value;
			trainfile >> value;
			examples[i].inputs.push_back(value);
		}
		for (int k = 0; k < numOutput; k++) {
			double value;
			trainfile >> value;
			examples[i].output.push_back(value);
		}
	}

	for (int i = 0; i < epoch; i++) {
		for (int j = 0; j < numTrainingExamples; j++) {
			// initialize inputs
			for (int k = 1; k <= numIns; k++) {
				inputLayer[k].activation = examples[j].inputs[k - 1];
			}
			// forward prop to hidden layer
			for (int m = 1; m <= numHidden; m++) {
				hiddenLayer[m].input = 0;
				for (vector<edge>::iterator it = hiddenLayer[m].inEdges.begin(); it != hiddenLayer[m].inEdges.end(); it++) {
					hiddenLayer[m].input += it->weight * it->toNode->activation;
				}
				hiddenLayer[m].activation = sigmoid(hiddenLayer[m].input);
			}

			// forward prop to output layer
			for (int c = 1; c <= numOutput; c++) {
				outputLayer[c].input = 0;
				for (vector<edge>::iterator it = outputLayer[c].inEdges.begin(); it != outputLayer[c].inEdges.end(); it++) {
					outputLayer[c].input += it->weight * it->toNode->activation;
				}
				outputLayer[c].activation = sigmoid(outputLayer[c].input);
			}

			// compute error
			for (int n = 1; n <= numOutput; n++) {
				outputLayer[n].error = sigmoidPrime(outputLayer[n].input) * (examples[j].output[n - 1] - outputLayer[n].activation);
			}
			for (int x = 1; x <= numHidden; x++) {
				double sum = 0;
				for (vector<edge>::iterator it = hiddenLayer[x].outEdges.begin(); it != hiddenLayer[x].outEdges.end(); it++) {
					sum += it->weight * it->toNode->error;
				}
				hiddenLayer[x].error = sigmoidPrime(hiddenLayer[x].input) * sum;
			}

			for (int a = 1; a <= numHidden; a++) {
				for (vector<edge>::iterator it = hiddenLayer[a].inEdges.begin(); it != hiddenLayer[a].inEdges.end(); it++) {
					it->weight += learningRate * it->toNode->activation * hiddenLayer[a].error;
					it->toNode->outEdges[a - 1].weight = it->weight;
				}
			}

			for (int a = 1; a <= numOutput; a++) {
				for (vector<edge>::iterator it = outputLayer[a].inEdges.begin(); it != outputLayer[a].inEdges.end(); it++) {
					it->weight += learningRate * it->toNode->activation * outputLayer[a].error;
					it->toNode->outEdges[a - 1].weight = it->weight;
				}
			}
		}
	}

	trainfile.close();
}

void nnets::test(ifstream& testFile, ostream& outFile) {
	int numTestExamples, numIns, numOuts;
	testFile >> numTestExamples >> numIns >> numOuts;  // 300 30 1

	struct testingExample {
		vector<double> inputs;
		vector<double> output;
	};
	vector<testingExample> examples;
	examples.resize(numTestExamples);

	double globalA = 0, globalB = 0, globalC = 0, globalD = 0;
	vector<double> contingencyTable = { 0,0,0,0 };
	vector<vector<double>> contingencyTables;
	for (int i = 0; i < numOuts; i++) {
		contingencyTables.push_back(contingencyTable);
	}

	for (int i = 0; i < numTestExamples; i++) {
		testingExample tmp;
		tmp.inputs.resize(numIns);
		tmp.output.resize(numOuts);
		for (int j = 0; j < numIns; j++) {
			double value;
			testFile >> value;
			examples[i].inputs.push_back(value);
		}
		for (int k = 0; k < numOutput; k++) {
			double value;
			testFile >> value;
			examples[i].output.push_back(value);
		}

		for (int j = 1; j <= numIns; j++) {
			inputLayer[j].activation = examples[i].inputs[j - 1];
		}
		for (int k = 1; k <= numHidden; k++) {
			hiddenLayer[k].input = 0;
			for (vector<edge>::iterator it = hiddenLayer[k].inEdges.begin(); it != hiddenLayer[k].inEdges.end(); it++) {
				hiddenLayer[k].input += it->weight * it->toNode->activation;
			}
			hiddenLayer[k].activation = sigmoid(hiddenLayer[k].input);
		}
		for (int c = 1; c <= numOutput; c++) {
			outputLayer[c].input = 0;
			for (vector<edge>::iterator it = outputLayer[c].inEdges.begin(); it != outputLayer[c].inEdges.end(); it++) {
				outputLayer[c].input += it->weight * it->toNode->activation;
			}
			outputLayer[c].activation = sigmoid(outputLayer[c].input);
		}

		for (int l = 1; l <= numOutput; l++) {
			if (outputLayer[l].activation >= 0.5) {
				if (examples[i].output[l - 1]) {
					contingencyTables[l - 1][0]++;
					globalA++;
				}
				else {
					contingencyTables[l - 1][1]++;
					globalB++;
				}
			}
			else {
				if (examples[i].output[l - 1]) {
					contingencyTables[l - 1][2]++;
					globalC++;
				}
				else {
					contingencyTables[l - 1][3]++;
					globalD++;
				}
			}
		}
	}
	double A, B, C, D, overall, precision, recall, f1;
	double macroOverall = 0, macroPrecision = 0, macroRecall = 0;
	outFile << setprecision(3) << fixed;
	for (int i = 0; i < numOuts; i++) {
		A = contingencyTables[i][0];
		B = contingencyTables[i][1];
		C = contingencyTables[i][2];
		D = contingencyTables[i][3];

		overall = (A + D) / (A + B + C + D);
		precision = A / (A + B);
		recall = A / (A + C);
		f1 = (2 * precision * recall) / (precision + recall);
		outFile << (int)A << " " << (int)B << " " << (int)C << " " << (int)D << " " << overall << " " << precision << " " << recall << " " << f1 << endl;

		macroOverall += overall;
		macroPrecision += precision;
		macroRecall += recall;
	}

	macroOverall /= numOuts;
	macroPrecision /= numOuts;
	macroRecall /= numOuts;
	double macroF1 = (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall);

	double microOverall = (globalA + globalD) / (globalA + globalB + globalC + globalD);
	double microPrecision = globalA / (globalA + globalB);
	double microRecall = globalA / (globalA + globalC);
	double microF1 = (2 * microPrecision * microRecall) / (microPrecision + microRecall);

	outFile << setprecision(3) << fixed << microOverall << " " << microPrecision << " " << microRecall << " " << microF1 << endl;
	outFile << setprecision(3) << fixed << macroOverall << " " << macroPrecision << " " << macroRecall << " " << macroF1 << endl;
}

void nnets::print(ostream& outfile) {
	outfile << setprecision(3) << fixed << numInput << " " << numHidden << " " << numOutput << endl;

	for (int i = 1; i <= numHidden; i++) {
		for (vector<edge>::iterator it = hiddenLayer[i].inEdges.begin(); it != hiddenLayer[i].inEdges.end(); it++) {
			if (it != hiddenLayer[i].inEdges.begin()) {
				outfile << " ";
			}
			outfile << it->weight;
		}
		outfile << endl;
	}

	for (int j = 1; j <= numOutput; j++) {
		for (vector<edge>::iterator it = outputLayer[j].inEdges.begin(); it != outputLayer[j].inEdges.end(); it++) {
			if (it != outputLayer[j].inEdges.begin()) {
				outfile << " ";
			}
			outfile << it->weight;
		}
		outfile << endl;
	}
}

double nnets::sigmoid(double value) {
	return 1 / (1 + exp(-value));
}

double nnets::sigmoidPrime(double value) {
	return (sigmoid(value) * (1 - sigmoid(value)));
}