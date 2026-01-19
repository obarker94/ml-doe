package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShift(t *testing.T) {
	t.Run("shifts all elements succesfully", func(t *testing.T) {
		// Arrange
		v := []float64{4.2, 3.1, 5.9}
		shift := 0.1

		// Act
		shifted, err := Shift(v, shift)

		// Assert
		assert.NoError(t, err)
		assert.InDeltaSlice(t, []float64{4.3, 3.2, 6.0}, shifted, 1e-9)
	})

	t.Run("returns an error if input vector has no length", func(t *testing.T) {
		// Arrange
		v := []float64{}
		shift := 0.1

		// Act
		_, err := Shift(v, shift)

		// Assert
		assert.Error(t, err)
	})

}
