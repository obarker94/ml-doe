package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBlock(t *testing.T) {
	t.Run("Block creates succesfully given valid LinearLayer and Nonlinearity", func(t *testing.T) {
		// Arrange
		block := Block{
			LinearLayer: LinearLayer{
				In:  2,
				Out: 2,
				W:   [][]float64{{4.3, -2.1}, {-9.8, 8.8}},
				B:   []float64{0.4, 2.2},
			},
			Nonlinearity: ReLU{},
		}

		// Act
		res, err := block.Forward([]float64{4.2, 4.2})

		// Assert
		assert.NoError(t, err)
		assert.InDeltaSlice(t, []float64{9.64, 0}, res, 1e-9)
	})

	t.Run("Block does not create if LinearLayer is invalid", func(t *testing.T) {
		// Arrange
		block := Block{
			LinearLayer:  LinearLayer{},
			Nonlinearity: ReLU{},
		}

		// Act
		_, err := block.Forward([]float64{4.2, 4.2})

		// Assert
		assert.Error(t, err)
	})

	t.Run("Block does not create if Nonlinearity is invalid", func(t *testing.T) {
		// Arrange
		block := Block{
			LinearLayer: LinearLayer{
				In:  2,
				Out: 2,
				W:   [][]float64{{4.3, -2.1}, {-9.8, 8.8}},
				B:   []float64{0.4, 2.2},
			},
			Nonlinearity: nil,
		}

		// Act
		_, err := block.Forward([]float64{4.2, 4.2})

		// Assert
		assert.Error(t, err)
	})

}
