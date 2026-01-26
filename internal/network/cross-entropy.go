package network

import (
	"fmt"
	"math"
)

func BinaryCrossEntropy(y, yHat float64) (float64, error) {
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0, fmt.Errorf("y is NaN/Inf")
	}
	if math.IsNaN(yHat) || math.IsInf(yHat, 0) {
		return 0, fmt.Errorf("yHat is NaN/Inf")
	}
	if y != 0 && y != 1 {
		return 0, fmt.Errorf("y must be 0 or 1, got %v", y)
	}

	// clamp yHat to avoid log(0)
	const eps = 1e-15
	if yHat < eps {
		yHat = eps
	}
	if yHat > 1-eps {
		yHat = 1 - eps
	}

	// -(y * log(yHat)) + (1-y)*log(1-yhat)
	return -(y*math.Log(yHat) + (1-y)*math.Log(1-yHat)), nil
}

func CrossEntropy(y, yHat []float64) (float64, error) {
	if len(y) == 0 {
		return 0, fmt.Errorf("y must have length")
	}
	if len(yHat) == 0 {
		return 0, fmt.Errorf("yHat must have length")
	}
	if len(y) != len(yHat) {
		return 0, fmt.Errorf("y and yHat must have same length")
	}

	const eps = 1e-15
	sumY, sumP := 0.0, 0.0
	loss := 0.0

	for i := range y {
		if y[i] != 0 && y[i] != 1 {
			return 0, fmt.Errorf("y must be one-hot, got y[%d]=%v", i, y[i])
		}
		sumY += y[i]

		p := yHat[i]
		if math.IsNaN(p) || math.IsInf(p, 0) {
			return 0, fmt.Errorf("yHat[%d] is NaN/Inf", i)
		}
		sumP += p

		if p < eps {
			p = eps
		} else if p > 1-eps {
			p = 1 - eps
		}

		loss -= y[i] * math.Log(p)
	}

	if math.Abs(sumY-1.0) > 1e-9 {
		return 0, fmt.Errorf("y must sum to 1, got %v", sumY)
	}
	if math.Abs(sumP-1.0) > 1e-6 {
		return 0, fmt.Errorf("yHat must sum to 1, got %v", sumP)
	}

	return loss, nil
}
