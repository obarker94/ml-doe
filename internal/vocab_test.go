package internal

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVocabCreation(t *testing.T) {
	t.Run("NewVocab instantiates succesfully if the corpus is correct", func(t *testing.T) {
		// Arrange
		corpus := "abcdefghijklmnopqrstuvwxyz1234567890 "

		// Act
		vocab, err := NewVocab([]rune(corpus))

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, len(vocab.symbols), len(corpus))
	})

	t.Run("NewVocab fails instantiate if corpus incorrect", func(t *testing.T) {
		// Arrange
		corpus := ""

		// Act
		vocab, err := NewVocab([]rune(corpus))

		// Assert
		assert.Error(t, err)
		assert.Nil(t, vocab)
	})

	t.Run("Duplication of runes inside 's' return an error", func(t *testing.T) {
		// Arrange
		corpus := "aabbcc"

		// Act
		vocab, err := NewVocab([]rune(corpus))

		// Assert
		assert.Error(t, err)
		assert.Nil(t, vocab)

	})
}

func TestVocabEncoding(t *testing.T) {
	corpus := "abcdefghijklmnopqrstuvwxyz1234567890£"
	vocab, err := NewVocab([]rune(corpus))
	assert.NoError(t, err)

	t.Run("Encode a string where the output has matching length of runes to ints", func(t *testing.T) {
		// Arrange
		example := "fizzbuzz"

		// Act
		ints, err := vocab.Encode(example)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, len(ints), len(example))
	})

	t.Run("Encode a string fails if attempting to encode an unknown rune", func(t *testing.T) {
		// Arrange
		example := "fizzbuzz " // contains space at the end, which is not in corpus.

		// Act
		ints, err := vocab.Encode(example)

		// Assert
		assert.Error(t, err)
		assert.NotEqual(t, len(ints), len(example))
	})

	t.Run("Encode a string with multibyte rune '£' should still return number of runes and not bytes", func(t *testing.T) {
		// Arrange
		example := "£100"

		// Act
		ints, err := vocab.Encode(example)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, len([]rune("£100")), len(ints))
	})
}

func TestVocabDecoding(t *testing.T) {
	corpus := "abcdefghijklmnopqrstuvwxyz1234567890£"
	vocab, err := NewVocab([]rune(corpus))
	assert.NoError(t, err)

	t.Run("Decode a slice of ints into a string if the ints can be mapped to the corpus", func(t *testing.T) {
		// Arrange
		example := []int{4, 6, 6}

		// Act
		result, err := vocab.Decode(example)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, "egg", result)
	})

	t.Run("Decode a slice of ints fails if an impossible int is looked up", func(t *testing.T) {
		// Arrange
		example := []int{4, 6, 6, 55}

		// Act
		result, err := vocab.Decode(example)

		// Assert
		assert.Error(t, err)
		assert.NotEqual(t, "egg", result)
	})

	t.Run("Decode a string with multibyte rune '£' be successful", func(t *testing.T) {
		// Arrange
		ints, err := vocab.Encode("£100")
		assert.NoError(t, err)

		// Act
		result, err := vocab.Decode(ints)

		// Assert
		assert.NoError(t, err)
		assert.Equal(t, "£100", result)
	})

}

func TestVocabRoundTrip(t *testing.T) {
	corpus := "abcdefghijklmnopqrstuvwxyz1234567890 \n.,[]{}()~!\"£$%^&*#"
	vocab, err := NewVocab([]rune(corpus))
	assert.NoError(t, err)

	t.Run("Encode and decode a string correctly", func(t *testing.T) {
		// Arrange
		example := "hello world.\n"

		// Act - Encode
		encoded, err := vocab.Encode(example)
		assert.NoError(t, err)

		// Act - Decode
		decoded, err := vocab.Decode(encoded)
		assert.NoError(t, err)

		// Assert
		assert.Equal(t, example, decoded)
	})
}
