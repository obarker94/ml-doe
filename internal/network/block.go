package network

import (
	"errors"
	"fmt"
)

func NewBlock(l LinearLayer, nl Nonlinearity) (*Block, error) {
	if err := l.Validate(); err != nil {
		return nil, errors.Join(errors.New("linearlayer failed validation"), err)
	}

	if nl == nil {
		return nil, fmt.Errorf("nonlinearity is nil and is required")
	}

	return &Block{
		LinearLayer:  l,
		Nonlinearity: nl,
	}, nil
}

type Block struct {
	LinearLayer  LinearLayer
	Nonlinearity Nonlinearity
}

func (b Block) Forward(x []float64) ([]float64, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}

	y, err := b.LinearLayer.Forward(x)
	if err != nil {
		return nil, errors.Join(errors.New("unable to create block"), err)
	}

	a, err := b.Nonlinearity.Apply(y)
	if err != nil {
		return nil, errors.Join(errors.New("unable to apply nonlinearity to block"), err)
	}

	return a, nil
}

func (b Block) Validate() error {
	if err := b.LinearLayer.Validate(); err != nil {
		return errors.Join(errors.New("block validation failed on linear layer"), err)
	}

	if b.Nonlinearity == nil {
		return fmt.Errorf("nonlinearity is nil and is required")
	}

	return nil
}
