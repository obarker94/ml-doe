package network

import (
	"fmt"
	"math"
)

// SoftmaxStats are the core stats to track probabilities, sums and saturation
// during a run.
type SoftmaxStats struct {
	Sum       float64
	SumProb   float64
	MaxLogit  float64
	MaxProb   float64
	MinProb   float64
	Saturated bool // e.g. MaxProb > 1-1e-12
}

const SaturationThreshold = 1 - 1e-5

// SoftmaxWithStats performs a Softmax on an input vector. It runs the resultant
// probability vector with stats.
func SoftmaxWithStats(z []float64) (p []float64, stats SoftmaxStats, err error) {
	if len(z) == 0 {
		return nil, SoftmaxStats{}, fmt.Errorf("input vector must have length greater than 0")

	}

	stats = SoftmaxStats{
		MaxLogit: z[0],
		MinProb:  math.Inf(1),
		MaxProb:  math.Inf(-1),
	}

	weights := make([]float64, len(z))
	probabilities := make([]float64, len(z))

	// Get the MaxLogit
	for idx, el := range z {
		if math.IsNaN(el) || math.IsInf(el, 0) {
			return nil, SoftmaxStats{}, fmt.Errorf("element %f position %d is invalid", el, idx)
		}
		if stats.MaxLogit > el {
			continue
		}
		stats.MaxLogit = el
	}

	// Calculate sum
	for idx, el := range z {
		weight := math.Exp(el - stats.MaxLogit)
		stats.Sum += weight
		weights[idx] = weight
	}

	if stats.Sum == 0 || math.IsNaN(stats.Sum) || math.IsInf(stats.Sum, 0) {
		return nil, SoftmaxStats{}, fmt.Errorf("softmax: invalid normalisation sum=%v", stats.Sum)
	}

	for i := range z {
		p := weights[i] / stats.Sum
		probabilities[i] = p

		stats.SumProb += p

		if p < stats.MinProb {
			stats.MinProb = p
		}
		if p > stats.MaxProb {
			stats.MaxProb = p
		}
	}
	stats.Saturated = stats.MaxProb >= SaturationThreshold

	return probabilities, stats, nil
}
