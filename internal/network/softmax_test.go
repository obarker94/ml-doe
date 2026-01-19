package network

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSoftmaxGolden(t *testing.T) {
	t.Run("z=[0,0] returns uniform distribution", func(t *testing.T) {
		// Arrange
		z := []float64{0, 0}
		expected := []float64{0.5, 0.5}

		// Act
		p, stats, err := SoftmaxWithStats(z)

		// Assert
		assert.NoError(t, err)
		assert.Len(t, p, 2)
		assert.InDeltaSlice(t, expected, p, 1e-9)

		assert.InDelta(t, 0.0, stats.MaxLogit, 1e-12)
		assert.InDelta(t, 0.5, stats.MaxProb, 1e-9)
		assert.InDelta(t, 0.5, stats.MinProb, 1e-9)
		assert.False(t, stats.Saturated)
	})

	t.Run("z=[1,0] returns expected", func(t *testing.T) {
		z := []float64{1, 0}
		expected := []float64{
			0.7310585786300049,
			0.2689414213699951,
		}

		p, stats, err := SoftmaxWithStats(z)

		assert.NoError(t, err)
		assert.Len(t, p, 2)
		assert.InDeltaSlice(t, expected, p, 1e-12)

		assert.InDelta(t, 1.0, stats.MaxLogit, 1e-12)
		assert.InDelta(t, expected[0], stats.MaxProb, 1e-9)
		assert.InDelta(t, expected[1], stats.MinProb, 1e-9)
		assert.False(t, stats.Saturated)
	})

	t.Run("z[-1000,-1000] returns expected", func(t *testing.T) {
		// Arrange
		z := []float64{-1000, -1000}
		expected := []float64{0.5, 0.5}

		// Act
		p, stats, err := SoftmaxWithStats(z)

		// Assert
		assert.NoError(t, err)
		assert.Len(t, p, 2)
		assert.InDeltaSlice(t, expected, p, 1e-9)

		assert.InDelta(t, -1000, stats.MaxLogit, 1e-9)
		assert.InDelta(t, 0.5, stats.MaxProb, 1e-12)
		assert.InDelta(t, 0.5, stats.MinProb, 1e-12)
		assert.False(t, stats.Saturated)
	})
}

func TestSoftmaxInvariants(t *testing.T) {
	t.Run("softmax with a constant is equal to softmax without", func(t *testing.T) {
		// Arrange
		z := []float64{1.4, 3.2, 8.8, 5.4}
		c := 598.23122
		zOffset := []float64{z[0] + c, z[1] + c, z[2] + c, z[3] + c}

		// Act
		p1, _, err := SoftmaxWithStats(z)
		assert.NoError(t, err)
		p2, _, err := SoftmaxWithStats(zOffset)
		assert.NoError(t, err)

		// Assert
		assert.InDeltaSlice(t, p1, p2, 1e-9)
	})

	t.Run("sum of probabilties ~= 1 within tolerance", func(t *testing.T) {
		// Arrange
		z := []float64{1.4, 3.2, 8.8, 5.4}

		// Act
		_, stats, err := SoftmaxWithStats(z)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, 1, stats.SumProb, 1e-12)
	})

	t.Run("number of logits is equal to number of probabilities", func(t *testing.T) {
		// Arrange
		z := []float64{1.4, 3.2, 8.8, 5.4}

		// Act
		p, _, err := SoftmaxWithStats(z)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, len(z), len(p))
	})
}

func TestSoftmaxErrorHandling(t *testing.T) {
	t.Run("errors if input vector is empty", func(t *testing.T) {
		// Arrange
		z := []float64{}

		// Act
		_, _, err := SoftmaxWithStats(z)

		// Assert
		assert.Error(t, err)
	})

	t.Run("errors if input vector contains -inf", func(t *testing.T) {
		// Arrange
		z := []float64{3.0, 4.0, math.Inf(-1)}

		// Act
		_, _, err := SoftmaxWithStats(z)

		// Assert
		assert.Error(t, err)
	})

	t.Run("errors if input vector contains inf", func(t *testing.T) {
		// Arrange
		z := []float64{3.0, 4.0, math.Inf(1)}

		// Act
		_, _, err := SoftmaxWithStats(z)

		// Assert
		assert.Error(t, err)
	})

	t.Run("errors if input vector contains NaN", func(t *testing.T) {
		// Arrange
		z := []float64{3.0, 4.0, math.NaN()}

		// Act
		_, _, err := SoftmaxWithStats(z)

		// Assert
		assert.Error(t, err)
	})
}
