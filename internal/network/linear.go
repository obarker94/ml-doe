package network

import (
	"errors"
	"fmt"

	"github.com/obarker94/ml-doe/internal"
)

type LinearLayer struct {
	In  int
	Out int
	W   [][]float64
	B   []float64
}

func (l *LinearLayer) Validate() error {
	if l.In <= 0 || l.Out <= 0 {
		return fmt.Errorf("invalid layer dims: In=%d Out=%d", l.In, l.Out)
	}

	if len(l.W) != l.Out {
		return fmt.Errorf("dimension mismatch: W has %d rows, expected %d", len(l.W), l.Out)
	}

	return nil

}

// Forward is both a validation and a Forward computation for given weights and
// input.
func (l *LinearLayer) Forward(x []float64) ([]float64, error) {
	if err := l.Validate(); err != nil {
		return nil, errors.Join(errors.New("Linearlayer failed validation"), err)
	}

	if len(x) != l.In {
		return nil, fmt.Errorf("dimension mismatch: x has length %d, expected %d", len(x), l.In)
	}

	withBias := false
	if len(l.B) != 0 {
		if len(l.B) != l.Out {
			return nil, fmt.Errorf("dimension mismatch: Bias length %d, expected equal length to out %d", len(l.B), l.Out)
		}
		withBias = true
	}

	output, err := internal.MatVecMul(l.W, x)
	if err != nil {
		return nil, errors.Join(errors.New("LinearLayer Forward failed at MatVecMul"), err)
	}

	if withBias {
		return internal.AddVec(output, l.B)
	}

	return output, nil
}
