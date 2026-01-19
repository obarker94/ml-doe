package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMLP(t *testing.T) {
	t.Run("logits argmax is negative shift-invariant", func(t *testing.T) {
		// Arrange
		mlp := MLP{
			Hidden: Block{
				LinearLayer: LinearLayer{
					In:  3,
					Out: 4,
					W:   [][]float64{{4.4, 3.2, 2.1}, {1.1, 2.1, 3.1}, {4.4, 3.2, 2.1}, {4.4, 3.2, 2.1}},
					B:   []float64{0.4, 0.7, 1.2, 3.3},
				},
				Nonlinearity: ReLU{},
			},
			Out: LinearLayer{
				In:  4,
				Out: 3,
				W:   [][]float64{{4.4, 3.2, 2.1, 0.4}, {1.1, 2.1, 3.1, 0.2}, {4.4, 3.2, 2.1, 0.5}},
				B:   []float64{0.4, 0.7, 1.2},
			},
		}

		// Act
		x := []float64{9.9, 8.8, 3.9}
		z, err := mlp.Forward(x)
		assert.NoError(t, err)

		shiftedZ, err := Shift(z, -500.2)
		assert.NoError(t, err)

		// Assert
		argMaxZ, err := ArgMax(z)
		assert.NoError(t, err)

		argMaxShiftedZ, err := ArgMax(shiftedZ)
		assert.NoError(t, err)

		assert.Equal(t, argMaxZ, argMaxShiftedZ)

	})

	t.Run("logits argmax is shift-invariant", func(t *testing.T) {
		// Arrange
		mlp := MLP{
			Hidden: Block{
				LinearLayer: LinearLayer{
					In:  3,
					Out: 4,
					W:   [][]float64{{4.4, 3.2, 2.1}, {1.1, 2.1, 3.1}, {4.4, 3.2, 2.1}, {4.4, 3.2, 2.1}},
					B:   []float64{0.4, 0.7, 1.2, 3.3},
				},
				Nonlinearity: ReLU{},
			},
			Out: LinearLayer{
				In:  4,
				Out: 3,
				W:   [][]float64{{4.4, 3.2, 2.1, 0.4}, {1.1, 2.1, 3.1, 0.2}, {4.4, 3.2, 2.1, 0.5}},
				B:   []float64{0.4, 0.7, 1.2},
			},
		}

		// Act
		x := []float64{9.9, 8.8, 3.9}
		z, err := mlp.Forward(x)
		assert.NoError(t, err)

		shiftedZ, err := Shift(z, 5.2)
		assert.NoError(t, err)

		// Assert
		argMaxZ, err := ArgMax(z)
		assert.NoError(t, err)

		argMaxShiftedZ, err := ArgMax(shiftedZ)
		assert.NoError(t, err)

		assert.Equal(t, argMaxZ, argMaxShiftedZ)

	})

	t.Run("MLP fails validation if block out is not equal to output layer in", func(t *testing.T) {
		// Arrange
		mlp := MLP{
			Hidden: Block{
				LinearLayer: LinearLayer{
					In:  3,
					Out: 4,
					W:   [][]float64{{4.4, 3.2, 2.1}, {1.1, 2.1, 3.1}, {4.4, 3.2, 2.1}, {4.4, 3.2, 2.1}},
					B:   []float64{0.4, 0.7, 1.2, 3.3},
				},
				Nonlinearity: ReLU{},
			},
			Out: LinearLayer{
				In:  5,
				Out: 3,
				W:   [][]float64{{4.4, 3.2, 2.1, 0.4, 2.1}, {1.1, 2.1, 3.1, 0.2, 3.1}, {4.4, 3.2, 2.1, 0.5, 2.1}},
				B:   []float64{0.4, 0.7, 1.2},
			},
		}

		// Act
		err := mlp.Validate()

		// Assert
		assert.Error(t, err)
	})

}
