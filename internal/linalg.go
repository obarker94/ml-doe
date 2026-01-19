package internal

import (
	"errors"
	"fmt"
)

// Dot is an expected Dot Product given two vectors.
func Dot(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0.0, fmt.Errorf("slices must be the same length")
	}

	output := 0.0
	for idx, i := range a {
		output += i * b[idx]
	}

	return output, nil
}

// MatVecMul is a matric vector multiplier with an optional bias vector
func MatVecMul(W [][]float64, x []float64) ([]float64, error) {
	// We must check that W is strictly rectangular.
	if err := isRectangular(W); err != nil {
		return nil, errors.Join(errors.New("MatVecMul unable to run"), err)
	}

	output := make([]float64, len(W))

	for idx, row := range W {
		if len(row) != len(x) {
			return nil, fmt.Errorf("dimension mismatch: W row %d has length %d, x has length %d", idx, len(row), len(x))
		}

		product, err := Dot(row, x)
		if err != nil {
			return nil, fmt.Errorf("MatVecMul: dot failed on row %d (row=%v, x=%v): %w", idx, row, x, err)
		}

		output[idx] = product
	}

	return output, nil
}

// AddVec takes in vector of length D and bias vector of equal length and then
// adds the bias to each scalar.
func AddVec(v []float64, b []float64) ([]float64, error) {
	if len(b) != len(v) {
		return nil, fmt.Errorf("bias length %d is not equal to input vector length %d", len(b), len(v))
	}

	output := make([]float64, len(v))

	for idx := range b {
		output[idx] = v[idx] + b[idx]
	}

	return output, nil
}

func isRectangular(m [][]float64) error {
	if len(m) == 0 {
		return fmt.Errorf("W must have at least one row")
	}

	// We check strict rectangular-ness by using the 0th elements width against
	// all rows.

	width := len(m[0])

	for idx, row := range m {
		if len(row) != width {
			return fmt.Errorf(
				"shape is not rectangular: row %d has length %d, expected %d",
				idx, len(row), width)
		}
	}

	return nil
}
