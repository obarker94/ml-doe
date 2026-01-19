package network

import (
	"errors"
	"fmt"
)

// MLP is a multi-layer perceptron. It contains hidden blocks that can be
// activated. The resultant vectors are consumed via the output linear layer.
type MLP struct {
	Hidden Block
	Out    LinearLayer
}

// Forward passes the MLP to pass Hidden block results to the output layer.
func (m MLP) Forward(x []float64) ([]float64, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}

	a, err := m.Hidden.Forward(x)
	if err != nil {
		return nil, errors.Join(errors.New("MLP unable to forward block"), err)
	}

	z, err := m.Out.Forward(a)
	if err != nil {
		return nil, errors.Join(errors.New("MLP unable to forward output layer"), err)
	}

	return z, nil
}

// Validate ensures that the Hidden blocks output is the same size as the
// Output layers input.
func (m MLP) Validate() error {
	if err := m.Hidden.Validate(); err != nil {
		return errors.Join(errors.New("MLP failed to validate hidden block"), err)
	}

	if err := m.Out.Validate(); err != nil {
		return errors.Join(errors.New("MLP failed to validate output layer"), err)
	}

	if m.Hidden.LinearLayer.Out != m.Out.In {
		return fmt.Errorf("dimension mismatch: block size %d and output layer input size %d", m.Hidden.LinearLayer.Out, m.Out.In)
	}

	return nil
}
