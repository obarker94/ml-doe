package internal

import (
	"errors"
	"fmt"
)

// Vocab is a container around symbols or tokens and their mapping to integers.
type Vocab struct {
	symbols []rune
	runeIds map[rune]int
}

func NewVocab(s []rune) (*Vocab, error) {
	if len(s) == 0 {
		return nil, errors.New("Length of runes must be greater than 0. Vocab cannot be constructed")
	}

	// We copy the inbound runes to prevent mutation out of our control
	safeRunes := append([]rune(nil), s...)
	mappedIds := make(map[rune]int, len(safeRunes))

	for idx, char := range safeRunes {
		// check that the char does not already exist
		_, exists := mappedIds[char]
		if exists {
			return nil, fmt.Errorf("duplicate rune detected in '%s'", string(safeRunes))
		}
		mappedIds[char] = idx

	}

	return &Vocab{
		symbols: safeRunes,
		runeIds: mappedIds,
	}, nil
}

// Encode takes in a string and attempts to encode into a slice of ints.
func (v *Vocab) Encode(s string) ([]int, error) {
	if len(s) == 0 {
		return nil, errors.New("Encode must receive a string of length greater than 0.")
	}

	encodedRunes := make([]int, 0, len([]rune(s)))

	runePos := 0
	for _, r := range s {
		encodedInt, ok := v.runeIds[r]
		if !ok {
			return nil, fmt.Errorf("rune '%s' at position '%d' not found in vocab '%s'", string(r), runePos, string(v.symbols))
		}
		encodedRunes = append(encodedRunes, encodedInt)
		runePos++
	}

	return encodedRunes, nil
}

// Decode receives a slice of token ints and attempts to map them to runes to output a string.
func (v *Vocab) Decode(token []int) (string, error) {
	if len(token) <= 0 {
		return "", errors.New("Decode must receive a token with length greater than 0.")
	}

	output := make([]rune, len(token))

	// For each element in token, we have an integer representing the value of the symbol
	// for example, the token could 2,3,4 which would be 'bcd'. The vocab symbols will
	// be a slice of whatever we enter, this is typically the alphabet. Thus, we can
	// use the token value as the idx to access on the symbols
	for idx, number := range token {
		if number < 0 || number >= len(v.symbols) {
			return "", fmt.Errorf("decoding out of range token number from the vocab. Number: %d, Symbols: %s, Symbols Length: %d", number, string(v.symbols), len(v.symbols))
		}
		foundRune := v.symbols[number]
		output[idx] = foundRune
	}

	return string(output), nil
}
