package network

import "fmt"

type ReLU struct{}

func (r ReLU) Apply(v []float64) ([]float64, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("v must not be length 0")
	}

	output := make([]float64, len(v))

	for idx, el := range v {
		output[idx] = max(0, el)
	}

	return output, nil
}
