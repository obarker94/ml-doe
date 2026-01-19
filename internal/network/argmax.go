package network

import "fmt"

// Returns the position of the largest element
func ArgMax(v []float64) (int, error) {
	if len(v) == 0 {
		return 0, fmt.Errorf("input vector must have length")
	}

	maxIdx := 0
	maxVal := v[0]

	for i := 1; i < len(v); i++ {
		if v[i] > maxVal {
			maxVal = v[i]
			maxIdx = i
		}
	}

	return maxIdx, nil
}
