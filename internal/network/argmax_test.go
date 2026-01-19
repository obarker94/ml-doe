package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestArgMax(t *testing.T) {
	t.Run("largest arg idx returned", func(t *testing.T) {
		// Arrange
		v := []float64{4.3, 3.1, 9.9, 2.2}

		// Act
		idx, err := ArgMax(v)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, 2, idx)
	})

	t.Run("returns error if input vector has no length", func(t *testing.T) {
		// Arrange
		v := []float64{}

		// Act
		_, err := ArgMax(v)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns idx of largest vector when there are negatives", func(t *testing.T) {
		// Arrange
		v := []float64{-4.4, -9.9, -2.4, -10.2, -0.4}

		// Act
		idx, err := ArgMax(v)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, 4, idx)
	})

}
