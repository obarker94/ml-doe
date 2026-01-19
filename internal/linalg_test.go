package internal

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDotProduct(t *testing.T) {
	t.Run("Dot returns the the product succesfully", func(t *testing.T) {
		// Arrange
		slice1 := []float64{0.0, 4.2, 3.4}
		slice2 := []float64{4.0, 3.3, 7.9}

		// Act
		product, err := Dot(slice1, slice2)

		// Assert
		assert.NoError(t, err)
		assert.InDelta(t, 40.72, product, 1e-9)
	})

	t.Run("Dot fails if slice1 is mismatch in size to slice 2", func(t *testing.T) {
		// Arrange
		slice1 := []float64{0.0, 4.2, 3.4}
		slice2 := []float64{4.0, 3.3, 7.9, 3.2}

		// Act
		_, err := Dot(slice1, slice2)

		// Assert
		assert.Error(t, err)
	})

	t.Run("Dot fails if slice2 is mismatch in size to slice 1", func(t *testing.T) {
		// Arrange
		slice1 := []float64{0.0, 4.2, 3.4, 4.4}
		slice2 := []float64{4.0, 3.3, 7.9}

		// Act
		_, err := Dot(slice1, slice2)

		// Assert
		assert.Error(t, err)
	})
}

func TestMatVecMul(t *testing.T) {
	t.Run("multiplies matrix by vector correctly", func(t *testing.T) {
		// Arrange
		W := [][]float64{
			{1, 2},
			{3, 4},
		}
		x := []float64{5, 6}

		expected := []float64{17, 39}

		// Act
		result, err := MatVecMul(W, x)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, expected, result)
	})

	t.Run("fails if a row dimension is not equivalent in size to x", func(t *testing.T) {
		// Arrange
		exampleX := []float64{0.0, 2.2, 3.3}
		exampleW := [][]float64{{3.2, 4.4, 5.5, 2.2}}

		// Act
		_, err := MatVecMul(exampleW, exampleX)

		// Assert
		assert.Error(t, err)
	})

	t.Run("fails if W has no rows", func(t *testing.T) {
		// Arrange
		exampleX := []float64{0.0, 2.2}

		// Act
		_, err := MatVecMul(nil, exampleX)

		// Assert
		assert.Error(t, err)

	})
}

func TestHelpers(t *testing.T) {
	t.Run("isRectangular returns success for a valid rectangular matrix", func(t *testing.T) {
		// Arrange
		example := [][]float64{
			{4.2, 3.4, 1.9},
			{3.2, 3.3, 2.1},
		}

		// Act
		err := isRectangular(example)

		// Assert
		assert.NoError(t, err)
	})

	t.Run("isRectangular returns error if matrix is size 0", func(t *testing.T) {
		// Arrange
		example := [][]float64{}

		// Act
		err := isRectangular(example)

		// Assert
		assert.Error(t, err)
	})

	t.Run("isRectangular returns error if matrix is not a rectangle", func(t *testing.T) {
		// Arrange
		example := [][]float64{
			{4.2, 3.4, 1.9},
			{3.2, 2.1},
			{3.2, 3.3, 2.1},
		}

		// Act
		err := isRectangular(example)

		// Assert
		assert.Error(t, err)
	})

	t.Run("AddVec adds two vectors succesfully", func(t *testing.T) {
		// Arrange
		v1 := []float64{0.0, 1.0, 1.5}
		v2 := []float64{0.1, 1.1, 2.1}

		// Act
		result, err := AddVec(v1, v2)

		// Assert
		assert.NoError(t, err)
		assert.InDeltaSlice(t, []float64{0.1, 2.1, 3.6}, result, 1e-9)
	})

	t.Run("AddVec fails if two vectors not identical length", func(t *testing.T) {
		// Arrange
		v1 := []float64{0.0, 1.0, 1.5, 3.2}
		v2 := []float64{0.1, 1.1, 2.1}

		// Act
		_, err := AddVec(v1, v2)

		// Assert
		assert.Error(t, err)

	})

}
