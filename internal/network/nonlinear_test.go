package network

import (
	"testing"

	"github.com/obarker94/ml-doe/internal"
	"github.com/stretchr/testify/assert"
)

func TestNonLinearContract(t *testing.T) {
	t.Run("Test ReLU", func(t *testing.T) {
		nlc := NewNonlinearityContract(ReLU{})
		nlc.Test(t)
	})
}

func TestReLU(t *testing.T) {
	relu := ReLU{}

	t.Run("Apply clamps the negative values to 0", func(t *testing.T) {
		// Arrange
		input := []float64{4.5, 9.9, 7.3, -9.2, -4.3}

		// Act
		res, err := relu.Apply(input)

		// Assert
		assert.NoError(t, err)
		assert.InDeltaSlice(t, []float64{4.5, 9.9, 7.3, 0.0, 0.0}, res, 1e-9)
	})

	t.Run("ReLU breaks additivity: ReLU(u+v) != ReLU(u)+ReLU(v)", func(t *testing.T) {
		// Arrange
		u := []float64{1.0}
		v := []float64{-2.0}

		uPlusV, err := internal.AddVec(u, v)
		assert.NoError(t, err)

		// Act
		reluUPlusV, err := relu.Apply(uPlusV)
		assert.NoError(t, err)

		reluU, err := relu.Apply(u)
		assert.NoError(t, err)

		reluV, err := relu.Apply(v)
		assert.NoError(t, err)

		reluUPlusReluV, err := internal.AddVec(reluU, reluV)
		assert.NoError(t, err)

		// Assert
		assert.NotEqual(t, reluUPlusV, reluUPlusReluV)
	})
}
