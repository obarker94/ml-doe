package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Nonlinearity is the public interface used for adding curvature to layers.
type Nonlinearity interface {
	// Function that applies element wise to a vector. The result will be of
	// same length and exists to transform the linear layer with some curvature.
	Apply(v []float64) ([]float64, error)
}

// The NonLinearityContract is to create a common testing of the interface as
// we extend to add other mathematical concepts whilst still preserving the
// integrity of the contract.
type NonLinearityContract struct {
	Nl Nonlinearity
}

func NewNonlinearityContract(nl Nonlinearity) *NonLinearityContract {
	return &NonLinearityContract{
		Nl: nl,
	}
}

// Test is a contract level method for top-level tests that must run and be
// common against all implementers of the interface.
func (n *NonLinearityContract) Test(t *testing.T) {
	t.Run("Apply returns an output vector of same length as input vector", func(t *testing.T) {
		// Arrange
		input := []float64{2.1, 3.4, 2.2}

		// Act
		result, err := n.Nl.Apply(input)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, len(input), len(result))
	})

	t.Run("Apply returns an error if the input vector is of length 0", func(t *testing.T) {
		// Arrange
		input := []float64{}

		// Act
		_, err := n.Nl.Apply(input)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Apply does not mutate the input slice", func(t *testing.T) {
		// Arrange
		input := []float64{2.1, 3.4, 2.2}
		safeInput := make([]float64, len(input))
		copy(safeInput, input)

		// Act
		res1, err := n.Nl.Apply(input)
		assert.NoError(t, err)
		res2, err := n.Nl.Apply(input)

		// Assert first
		assert.NoError(t, err)
		assert.InDeltaSlice(t, safeInput, input, 1e-9)
		assert.InDeltaSlice(t, res1, res2, 1e-9)
	})

}
