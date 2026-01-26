package network

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBinaryCrossEntropy(t *testing.T) {
	t.Run("y=1 uses -log(yHat)", func(t *testing.T) {
		// Arrange
		y := 1.0
		yHat := 0.9
		expected := -math.Log(0.9)

		// Act
		loss, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, expected, loss, 1e-12)
	})

	t.Run("y=0 uses -log(1-yHat)", func(t *testing.T) {
		// Arrange
		y := 0.0
		yHat := 0.9
		expected := -math.Log(1.0 - 0.9)

		// Act
		loss, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, expected, loss, 1e-12)
	})

	t.Run("punishes confident wrong predictions more than timid ones", func(t *testing.T) {
		// Arrange
		y := 1.0
		confidentWrong := 0.01
		timid := 0.51

		// Act
		lossConfidentWrong, err1 := BinaryCrossEntropy(y, confidentWrong)
		lossTimid, err2 := BinaryCrossEntropy(y, timid)

		// Assert
		assert.NoError(t, err1)
		assert.NoError(t, err2)
		assert.Greater(t, lossConfidentWrong, lossTimid)
	})

	t.Run("returns error if y is not 0 or 1", func(t *testing.T) {
		// Arrange
		y := 0.2
		yHat := 0.8

		// Act
		_, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if y is NaN", func(t *testing.T) {
		// Arrange
		y := math.NaN()
		yHat := 0.8

		// Act
		_, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if yHat is NaN", func(t *testing.T) {
		// Arrange
		y := 1.0
		yHat := math.NaN()

		// Act
		_, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("clamps yHat=0 to avoid log(0) and returns a finite loss", func(t *testing.T) {
		// Arrange
		y := 1.0
		yHat := 0.0

		// Act
		loss, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.False(t, math.IsInf(loss, 0))
		assert.False(t, math.IsNaN(loss))
		// With eps=1e-15, expected is -log(1e-15)
		assert.InDelta(t, -math.Log(1e-15), loss, 1e-12)
	})

	t.Run("clamps yHat=1 to avoid log(0) and returns a finite loss", func(t *testing.T) {
		// Arrange
		y := 0.0
		yHat := 1.0
		const eps = 1e-15

		// What the function does (including float rounding)
		clamped := 1.0 - eps
		expected := -math.Log(1.0 - clamped) // == -log(effective eps after rounding)

		// Act
		loss, err := BinaryCrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.False(t, math.IsInf(loss, 0))
		assert.False(t, math.IsNaN(loss))
		assert.InDelta(t, expected, loss, 1e-12)
	})
}

func TestCrossEntropy(t *testing.T) {
	t.Run("returns -log(prob of true class) for one-hot y", func(t *testing.T) {
		// Arrange
		y := []float64{0, 1, 0}
		yHat := []float64{0.7, 0.2, 0.1}
		expected := -math.Log(0.2)

		// Act
		loss, err := CrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, expected, loss, 1e-12)
	})

	t.Run("lower loss when model assigns higher probability to true class", func(t *testing.T) {
		// Arrange
		y := []float64{0, 1, 0}
		yHatGood := []float64{0.05, 0.90, 0.05}
		yHatBad := []float64{0.90, 0.05, 0.05}

		// Act
		lossGood, err1 := CrossEntropy(y, yHatGood)
		lossBad, err2 := CrossEntropy(y, yHatBad)

		// Assert
		assert.NoError(t, err1)
		assert.NoError(t, err2)
		assert.Less(t, lossGood, lossBad)
	})

	t.Run("returns error if y has no length", func(t *testing.T) {
		// Arrange
		y := []float64{}
		yHat := []float64{1.0}

		// Act
		_, err := CrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if yHat has no length", func(t *testing.T) {
		// Arrange
		y := []float64{1.0}
		yHat := []float64{}

		// Act
		_, err := CrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if y and yHat lengths mismatch", func(t *testing.T) {
		// Arrange
		y := []float64{0, 1, 0}
		yHat := []float64{0.2, 0.8}

		// Act
		_, err := CrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if y is not one-hot (contains values other than 0 or 1)", func(t *testing.T) {
		// Arrange
		y := []float64{0, 0.5, 0.5}
		yHat := []float64{0.2, 0.3, 0.5}

		// Act
		_, err := CrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("returns error if y does not sum to 1", func(t *testing.T) {
		// Arrange
		y := []float64{1, 1, 0}
		yHat := []float64{0.2, 0.7, 0.1}

		// Act
		_, err := CrossEntropy(y, yHat)

		// Assert
		assert.Error(t, err)
	})

	t.Run("clamps yHat=0 at true class to avoid log(0) and returns finite loss", func(t *testing.T) {
		// Arrange
		y := []float64{0, 1, 0}
		yHat := []float64{0.5, 0.0, 0.5}

		// Act
		loss, err := CrossEntropy(y, yHat)

		// Assert
		assert.NoError(t, err)
		assert.False(t, math.IsInf(loss, 0))
		assert.False(t, math.IsNaN(loss))
		// expected is -log(eps) since true class prob clamps to eps
		assert.InDelta(t, -math.Log(1e-15), loss, 1e-12)
	})

	t.Run("ignores non-true classes because y is one-hot", func(t *testing.T) {
		// Arrange
		y := []float64{0, 1, 0}
		yHatA := []float64{0.7, 0.2, 0.1}
		yHatB := []float64{0.0, 0.2, 0.8} // same true-class prob (0.2), others changed

		// Act
		lossA, err1 := CrossEntropy(y, yHatA)
		lossB, err2 := CrossEntropy(y, yHatB)

		// Assert
		assert.NoError(t, err1)
		assert.NoError(t, err2)
		assert.InDelta(t, lossA, lossB, 1e-12)
	})
}
