package internal

import (
	"fmt"
)

type LinearLayer struct {
	In  int
	Out int
	W   [][]float64
}

func (l *LinearLayer) Forward(x []float64) ([]float64, error) {
	if l.In <= 0 || l.Out <= 0 {
		return nil, fmt.Errorf("invalid layer dims: In=%d Out=%d", l.In, l.Out)
	}

	if len(x) != l.In {
		return nil, fmt.Errorf("dimension mismatch: x has length %d, expected %d", len(x), l.In)
	}

	if len(l.W) != l.Out {
		return nil, fmt.Errorf("dimension mismatch: W has %d rows, expected %d", len(l.W), l.Out)
	}

	for i := range l.W {
		if len(l.W[i]) != l.In {
			return nil, fmt.Errorf("dimension mismatch: W row %d has length %d, expected %d", i, len(l.W[i]), l.In)
		}
	}

	return MatVecMul(l.W, x)
}

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

func MatVecMul(W [][]float64, x []float64) ([]float64, error) {
	if len(W) == 0 {
		return nil, fmt.Errorf("W must have at least one row")
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
