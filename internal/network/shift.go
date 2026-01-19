package network

import "fmt"

// Shift each element in a vector by amount x
func Shift(v []float64, x float64) ([]float64, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("input vector must have length")
	}

	shifted := make([]float64, len(v))
	for idx, el := range v {
		shifted[idx] = el + x
	}

	return shifted, nil
}
