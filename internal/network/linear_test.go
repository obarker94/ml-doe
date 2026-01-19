package network

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLinearLayer(t *testing.T) {
	t.Run("Forward runs successfully with valid input and no bias", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{1.0, 2.3, 4.2},
				{3.2, 0.0, 9.9},
			},
		}
		x := []float64{10, 1, -2}

		// Act
		got, err := ll.Forward(x)

		// Assert
		assert.NoError(t, err)
		assert.Len(t, got, 2)
		assert.InDelta(t, 3.9, got[0], 1e-9)
		assert.InDelta(t, 12.2, got[1], 1e-9)
	})

	t.Run("Forward returns bias with zero W", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{0.0, 0.0, 0.0},
				{0.0, 0.0, 0.0},
			},
			B: []float64{4.2, 3.0},
		}
		x := []float64{3.2, 1.2, 4.4}

		// Act
		got, err := ll.Forward(x)

		// Assert
		assert.NoError(t, err)
		assert.Len(t, got, 2)
		assert.InDelta(t, 4.2, got[0], 1e-9)
		assert.InDelta(t, 3.0, got[1], 1e-9)
	})

	t.Run("Forward runs successfully with valid bias and blank X", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 3,
			W: [][]float64{
				{3.2, 1.2, 3.9},
				{5.3, 3.1, 9.6},
				{5.7, 5.4, 8.6},
			},
			B: []float64{4.2, 3.0, 1.3},
		}
		x := []float64{0.0, 0.0, 0.0}

		// Act
		got, err := ll.Forward(x)

		// Assert
		assert.NoError(t, err)
		assert.Len(t, got, 3)
		assert.InDelta(t, 4.2, got[0], 1e-9)
		assert.InDelta(t, 3.0, got[1], 1e-9)
		assert.InDelta(t, 1.3, got[2], 1e-9)
	})

	t.Run("Forward runs fails with invalid bias different length to W (out)", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 3,
			W: [][]float64{
				{3.2, 1.2, 3.9},
				{5.3, 3.1, 9.6},
				{5.7, 5.4, 8.6},
			},
			B: []float64{4.2, 3.0, 1.3, 5.2},
		}
		x := []float64{0.0, 0.0, 0.0}

		// Act
		_, err := ll.Forward(x)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Forward fails when input dimension mismatches In", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
		}
		x := []float64{1, 2} // wrong length

		// Act
		_, err := ll.Forward(x)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Forward fails when W row count mismatches Out", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{1, 2, 3}, // only one row
			},
		}
		x := []float64{1, 2, 3}

		// Act
		_, err := ll.Forward(x)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Forward fails when a W row has wrong length", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{1, 2, 3},
				{4, 5}, // invalid row
			},
		}
		x := []float64{1, 2, 3}

		// Act
		_, err := ll.Forward(x)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Forward fails when layer dimensions are invalid", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  0,
			Out: 2,
			W:   [][]float64{},
		}
		x := []float64{}

		// Act
		_, err := ll.Forward(x)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Forward produces identical outputs for identical rows", func(t *testing.T) {
		// Arrange
		ll := LinearLayer{
			In:  3,
			Out: 2,
			W: [][]float64{
				{1, 2, 3},
				{1, 2, 3},
			},
		}
		x := []float64{4, 5, 6}

		// Act
		y, err := ll.Forward(x)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, y[0], y[1], 1e-9)
	})
}
