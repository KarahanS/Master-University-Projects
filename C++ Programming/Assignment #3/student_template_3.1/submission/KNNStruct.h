/*
Oguz Ata Cal    - 6661014  
Karahan Saritas - 6661689
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>


struct DYNPoint {
	std::vector<float> data;

	// We use a static function to return a DYNPoint object (this pattern is sometimes called a "factory method")
	// We can mark functions as nodiscard to indicate that we must use the return value in some way (e.g, it doesn't get discarded as ).
	[[nodiscard]] static DYNPoint createRandomPoint(unsigned int size, int minimum=-5000, int maximum=5000) {
		DYNPoint p;
		if (size > 0 && minimum <= maximum) {			
			p.data.resize(size);
			for (unsigned int i = 0; i < size; i++) {
				float j = float(rand()) / float(RAND_MAX);
				p.data[i] = float(minimum) + j * float(maximum - minimum);
			}		
		}
		return p;
	}
};

struct KNN {
	KNN(float(*_function_ptr_Distance)(const DYNPoint &, const DYNPoint &))
		: function_ptr_Distance(_function_ptr_Distance)
	{
	}

	std::vector<std::pair<DYNPoint, unsigned int>> trainingData{}; // default initialize on creation

	float(*function_ptr_Distance)(const DYNPoint &, const DYNPoint &) = 0;

	// just sets the training data (no actual training required)
	void TrainKNN(const std::vector<std::pair<DYNPoint, unsigned int>> &dataset) {
		trainingData.clear();
		trainingData = dataset;
	}

	int classify(const unsigned int k, const DYNPoint &A) const {

		int class_label = -1;

		if (k && function_ptr_Distance && trainingData.size()) {
			std::vector<std::pair<float, unsigned int>> distance_label;
			for (const auto &pair : trainingData) {
				distance_label.push_back(std::make_pair(function_ptr_Distance(A, pair.first), pair.second));
			}
			std::sort(distance_label.begin(), distance_label.end(), [](const std::pair<float, unsigned int> &a, const std::pair<float, unsigned int> &b) {
				return a.first < b.first;
			});
			std::unordered_map<unsigned int, unsigned int> class_freq;
			for (unsigned int i = 0; i < k; i++) {
				class_freq[distance_label[i].second]++;
			}

			// find the class with the highest frequency
			unsigned int max_freq = 0;
			for (const auto &pair : class_freq) {
				if (pair.second > max_freq) {
					max_freq = pair.second;
					class_label = pair.first;
				}
			}
		}
		return class_label;
	}
};

float DistanceManhattan(const DYNPoint &A, const DYNPoint &B) {
	int size = A.data.size();
	float distance = 0;
	for (int i = 0; i < size; i++) { 
		distance += std::abs(A.data[i] - B.data[i]);
	}
	return distance / size;
}

float DistanceEuclid(const DYNPoint &A, const DYNPoint &B) {
	unsigned int size = A.data.size();
	float distance = 0;
	for (unsigned int i = 0; i < size; i++) {
		distance += std::pow(A.data[i] - B.data[i], 2);
	}
	return std::sqrt(distance / size);
}

void createDataset(std::vector<std::pair<DYNPoint, unsigned int>> &dataset, const unsigned int amount, const unsigned int class_label,
				const unsigned int point_size, const int minimum, const int maximum) {

	if (amount > 0 && point_size>0 && minimum <= maximum) {
		for (unsigned int i = 0; i < amount; i++) {
			DYNPoint p = DYNPoint::createRandomPoint(point_size, minimum, maximum);
			dataset.push_back(std::make_pair(p, class_label));
		}

	}
}

void evaluateKNN(const std::vector<std::pair<DYNPoint, unsigned int>> &dataset, const KNN &Classifier, const unsigned int k) {
	if (!dataset.empty()) {
		float acc = 0;
		for (size_t i = 0; i < dataset.size(); i++) {
			if (static_cast<unsigned int>(Classifier.classify(k, dataset[i].first)) == dataset[i].second)
				acc++;
		}
		std::cout << "Accuracy: " << acc / float(dataset.size()) << std::endl;
	}
}









