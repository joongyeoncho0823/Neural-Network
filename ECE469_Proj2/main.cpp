#include "nnets.h"

using namespace std;

int main()
{
	string outFileName;
	ofstream outFile;

	int choice;

	cout << "[1]: Train\n[2] Test" << endl;
	cin >> choice;

	if (choice == 1)
	{
		double learningRate;
		int epoch;
		string initFileName, trainFileName;
		cout << "Name of initial neural network file" << endl;
		cin >> initFileName;
		cout << "Name of train file name" << endl;
		cin >> trainFileName;
		cout << "Name of output file name" << endl;
		cin >> outFileName;
		cout << "Input the number of epochs" << endl;
		cin >> epoch;
		cout << "Input the learning rate" << endl;
		cin >> learningRate;

		ifstream initFile, trainFile;
		initFile.open(initFileName);
		trainFile.open(trainFileName);
		outFile.open(outFileName);

		nnets test = nnets(initFile);
		test.train(trainFile, epoch, learningRate);
		test.print(outFile);
	}
	if (choice == 2)
	{
		string trainedFileName, testFileName;
		cout << "Name of trained neural network file" << endl;
		cin >> trainedFileName;
		cout << "Name of test file name" << endl;
		cin >> testFileName;
		cout << "Name of output file name" << endl;
		cin >> outFileName;

		ifstream trainedFile, testFile;
		trainedFile.open(trainedFileName);
		testFile.open(testFileName);
		outFile.open(outFileName);

		nnets test = nnets(trainedFile);
		test.test(testFile, outFile);
	}
	return 0;
}