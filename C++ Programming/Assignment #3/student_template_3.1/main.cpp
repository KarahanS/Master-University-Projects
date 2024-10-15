
#include <iostream>
#include <vector>

#include "submission/KNNStruct.h"

int main()
{
	DYNPoint A, B, C;

	A = DYNPoint::createRandomPoint(50, -5000, 5000);
	B = DYNPoint::createRandomPoint(50, -5000, 5000);
	C = DYNPoint::createRandomPoint(50, -5000, 5000);
	
	std::vector<std::pair<DYNPoint, unsigned int>> trainA, testA;

	createDataset(trainA, 100, 1, 50, -5000, 5000);
	createDataset(trainA, 100, 2, 50, -10000, 10000);
	createDataset(trainA, 100, 3, 50,  5000, 15000);

	createDataset(testA, 1000, 1, 50, -5000, 5000);
	createDataset(testA, 1000, 2, 50, -10000, 10000);
	createDataset(testA, 1000, 3, 50, 5000, 15000);

	KNN KNNA{&DistanceManhattan};
	KNN KNNB{&DistanceEuclid};

	KNNA.TrainKNN(trainA);
	KNNB.TrainKNN(trainA);

	for (unsigned int i = 1; i <= 10; i++) {
		std::cout << "K: " << i << " train error" << std::endl;
		evaluateKNN(trainA, KNNA, i);
		evaluateKNN(trainA, KNNB, i);
		
		std::cout << "K: " << i << " test error" << std::endl;
		evaluateKNN(testA, KNNA, i);
		evaluateKNN(testA, KNNB, i);
	}
	
	return 0;
}
