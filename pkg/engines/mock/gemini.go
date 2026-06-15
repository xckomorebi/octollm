package mock

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/infinigence/octollm/pkg/errutils"
	"github.com/infinigence/octollm/pkg/octollm"
	"github.com/infinigence/octollm/pkg/types/vertex"
)

type GeminiMockEndpoint struct {
	OutputString string
	TTFT         time.Duration
	TPOT         time.Duration
}

var _ octollm.Engine = (*GeminiMockEndpoint)(nil)

func NewGeminiWithFixedOutput(outputString string, ttft, tpot time.Duration) *GeminiMockEndpoint {
	return &GeminiMockEndpoint{
		OutputString: outputString,
		TTFT:         ttft,
		TPOT:         tpot,
	}
}

// See mockParams doc comment on the OpenAI MockEndpoint for parameter details.
// Gemini uses finishReason "STOP" / "MAX_TOKENS" and reports usage via
// UsageMetadata (PromptTokenCount / CandidatesTokenCount / CachedContentTokenCount).
//
// In addition, when the "image" param is truthy (e.g. image=1), the response
// contains only an inlineData part holding a base64 grayscale JPEG that spells
// "OCTOLLM" (no text part), computed in memory.
type geminiMockParams struct {
	rawParams    string
	httpStatus   int
	ttft         time.Duration
	tpot         time.Duration
	output       string
	outputRunes  []rune
	finishReason string
	usage        *vertex.UsageMetadata
	image        bool
}

func (e *GeminiMockEndpoint) newGeminiMockParams(v *vertex.GenerateContentRequest) *geminiMockParams {
	p := &geminiMockParams{
		ttft:       e.TTFT,
		tpot:       e.TPOT,
		httpStatus: 200,
	}

	maxTokens := 0
	if v.GenerationConfig != nil && v.GenerationConfig.MaxOutputTokens != nil {
		maxTokens = *v.GenerationConfig.MaxOutputTokens
	}

	outputLen := len([]rune(e.OutputString))
	cachedLen := 0
	inputLen := 0

	totalRunes := 0
	if len(v.Contents) > 0 {
		lastMsg := v.Contents[len(v.Contents)-1]
		firstContent := extractGeminiText(&lastMsg)
		lines := strings.SplitN(firstContent, "\n", 2)
		firstLine := strings.TrimSpace(lines[0])

		if strings.Contains(firstLine, "=") || strings.Contains(firstLine, "&") {
			p.rawParams = firstLine
			vals, err := url.ParseQuery(firstLine)
			if err == nil {
				if v := vals.Get("ttft"); v != "" {
					if d, err := time.ParseDuration(v); err == nil {
						p.ttft = d
					}
				}
				if v := vals.Get("tpot"); v != "" {
					if d, err := time.ParseDuration(v); err == nil {
						p.tpot = d
					}
				}
				if v := vals.Get("output_len"); v != "" {
					if n, err := strconv.Atoi(v); err == nil {
						outputLen = n
					}
				}
				if v := vals.Get("cached_len"); v != "" {
					if n, err := strconv.Atoi(v); err == nil {
						cachedLen = n
					}
				}
				if v := vals.Get("input_len"); v != "" {
					if n, err := strconv.Atoi(v); err == nil {
						inputLen = n
					}
				}
				if v := vals.Get("http_status"); v != "" {
					if n, err := strconv.Atoi(v); err == nil {
						p.httpStatus = n
					}
				}
				if v := vals.Get("image"); v != "" {
					switch strings.ToLower(v) {
					case "false":
					default:
						p.image = true
					}
				}
			}
			if len(lines) > 1 {
				totalRunes += len([]rune(lines[1]))
			}
		} else {
			totalRunes += len([]rune(firstContent))
		}

		for _, m := range v.Contents[:len(v.Contents)-1] {
			totalRunes += len([]rune(extractGeminiText(&m)))
		}
	}

	if v.SystemInstruction != nil {
		totalRunes += len([]rune(extractGeminiText(v.SystemInstruction)))
	}

	if inputLen == 0 {
		inputLen = totalRunes
	}

	if outputLen == 0 {
		p.output = ""
		p.outputRunes = nil
		p.finishReason = "STOP"
	} else {
		runes := []rune(e.OutputString)
		if len(runes) == 0 {
			runes = []rune(" ")
		}

		output := make([]rune, 0, outputLen)
		for len(output) < outputLen {
			remaining := outputLen - len(output)
			if remaining >= len(runes) {
				output = append(output, runes...)
			} else {
				output = append(output, runes[:remaining]...)
			}
		}

		p.finishReason = "STOP"
		if maxTokens > 0 && maxTokens < outputLen {
			output = output[:maxTokens]
			p.finishReason = "MAX_TOKENS"
		}

		p.output = string(output)
		p.outputRunes = output
	}

	p.usage = &vertex.UsageMetadata{
		PromptTokenCount:     inputLen,
		CandidatesTokenCount: len(p.outputRunes),
		TotalTokenCount:      inputLen + len(p.outputRunes),
	}
	if cachedLen > 0 {
		p.usage.CachedContentTokenCount = cachedLen
	}

	return p
}

func (p *geminiMockParams) String() string {
	return p.rawParams
}

func extractGeminiText(c *vertex.Content) string {
	if c == nil {
		return ""
	}
	var sb strings.Builder
	for _, part := range c.Parts {
		sb.WriteString(part.Text)
	}
	return sb.String()
}

func (e *GeminiMockEndpoint) Process(req *octollm.Request) (*octollm.Response, error) {
	reqBody, err := req.Body.Parsed()
	if err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	switch v := reqBody.(type) {
	case *vertex.GenerateContentRequest:
		p := e.newGeminiMockParams(v)
		slog.InfoContext(req.Context(), "[gemini-mock-endpoint] params: "+p.String())
		if p.httpStatus != 200 {
			time.Sleep(p.ttft)
			return nil, &errutils.UpstreamRespError{
				StatusCode: p.httpStatus,
				Header: http.Header{
					"Content-Type":  {"application/json"},
					"X-Mock-Params": {p.String()},
				},
				Body: []byte(fmt.Sprintf(`{"error":{"code":%d,"message":"mock error","status":"MOCK_ERROR"}}`, p.httpStatus)),
			}
		}

		// Gemini selects streaming by action (streamGenerateContent), not a body field.
		isStream := false
		if action, ok := octollm.GetCtxValue[string](req, octollm.ContextKeyAction); ok {
			isStream = octollm.IsStreamAction(action)
		}
		if isStream {
			return e.geminiStreamResponse(req, v, p)
		}
		return e.geminiNonStreamResponse(req, v, p)
	default:
		return nil, fmt.Errorf("unexpected request body type: %T", reqBody)
	}
}

func (e *GeminiMockEndpoint) modelVersion(req *octollm.Request) string {
	if model, ok := octollm.GetCtxValue[string](req, octollm.ContextKeyModelName); ok && model != "" {
		return model
	}
	return "mock"
}

func (e *GeminiMockEndpoint) geminiNonStreamResponse(req *octollm.Request, v *vertex.GenerateContentRequest, p *geminiMockParams) (*octollm.Response, error) {
	time.Sleep(p.ttft + p.tpot*time.Duration(len(p.outputRunes)))

	imgPart, err := p.imagePart()
	if err != nil {
		return nil, fmt.Errorf("failed to render mock image: %w", err)
	}
	// When an image is requested, return only the image (no text part).
	var parts []vertex.Part
	if imgPart != nil {
		parts = []vertex.Part{*imgPart}
	} else {
		parts = []vertex.Part{{Text: p.output}}
	}

	bodyVal := &vertex.GenerateContentResponse{
		Candidates: []vertex.Candidate{
			{
				Content: &vertex.Content{
					Role:  "model",
					Parts: parts,
				},
				FinishReason: p.finishReason,
			},
		},
		UsageMetadata: p.usage,
		ModelVersion:  e.modelVersion(req),
		ResponseID:    "mock-id",
	}
	body := octollm.NewBodyFromParsed(bodyVal, &octollm.JSONParser[vertex.GenerateContentResponse]{})
	hdr := http.Header{
		"Content-Type":  {"application/json"},
		"X-Mock-Params": {p.String()},
	}
	resp := octollm.NewNonStreamResponse(200, hdr, body)
	return resp, nil
}

func (e *GeminiMockEndpoint) geminiStreamResponse(req *octollm.Request, v *vertex.GenerateContentRequest, p *geminiMockParams) (*octollm.Response, error) {
	ch := make(chan *octollm.StreamChunk)
	ctx, cancel := context.WithCancel(req.Context())

	modelVersion := e.modelVersion(req)

	octollm.SafeGo(req, func() {
		defer close(ch)

		time.Sleep(p.ttft)

		// When an image is requested, stream only the image (no text chunks).
		textRunes := p.outputRunes
		if p.image {
			textRunes = nil
		}

		for i, c := range textRunes {
			chunk := &vertex.StreamGenerateContentResponse{
				Candidates: []vertex.Candidate{
					{
						Content: &vertex.Content{
							Role:  "model",
							Parts: []vertex.Part{{Text: string(c)}},
						},
					},
				},
				ModelVersion: modelVersion,
				ResponseID:   "mock-id",
			}
			// The last text chunk carries the finish reason and usage, matching Gemini.
			if i == len(textRunes)-1 {
				chunk.Candidates[0].FinishReason = p.finishReason
				chunk.UsageMetadata = p.usage
			}
			select {
			case ch <- &octollm.StreamChunk{
				Body: octollm.NewBodyFromParsed(chunk, &octollm.JSONParser[vertex.StreamGenerateContentResponse]{}),
			}:
			case <-ctx.Done():
				slog.InfoContext(ctx, fmt.Sprintf("[gemini-mock-endpoint] context canceled during stream response: %v", ctx.Err()))
				return
			}
			time.Sleep(p.tpot)
		}

		// Build the terminal chunk: an inline image when requested, otherwise an
		// empty content part. It carries the finish reason and usage. When there
		// was streamed text and no image, the loop above already finished.
		if p.image || len(p.outputRunes) == 0 {
			parts := []vertex.Part{{Text: ""}}
			if imgPart, err := p.imagePart(); err != nil {
				slog.ErrorContext(ctx, fmt.Sprintf("[gemini-mock-endpoint] failed to render mock image: %v", err))
			} else if imgPart != nil {
				parts = []vertex.Part{*imgPart}
			}
			chunk := &vertex.StreamGenerateContentResponse{
				Candidates: []vertex.Candidate{
					{
						Content:      &vertex.Content{Role: "model", Parts: parts},
						FinishReason: p.finishReason,
					},
				},
				UsageMetadata: p.usage,
				ModelVersion:  modelVersion,
				ResponseID:    "mock-id",
			}
			select {
			case ch <- &octollm.StreamChunk{
				Body: octollm.NewBodyFromParsed(chunk, &octollm.JSONParser[vertex.StreamGenerateContentResponse]{}),
			}:
			case <-ctx.Done():
				return
			}
		}
	})

	streamChan := octollm.NewStreamChan(ch, cancel)
	hdr := http.Header{
		"Content-Type":  {"text/event-stream"},
		"X-Mock-Params": {p.String()},
	}
	resp := octollm.NewStreamResponse(200, hdr, streamChan)
	return resp, nil
}

// imagePart returns an inlineData part holding a base64 grayscale JPEG that
// spells "OCTOLLM", or nil when no image was requested.
func (p *geminiMockParams) imagePart() (*vertex.Part, error) {
	if !p.image {
		return nil, nil
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, renderText("OCTOLLM", 256, 256), &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return &vertex.Part{
		InlineData: &vertex.Blob{
			MimeType: "image/jpeg",
			Data:     base64.StdEncoding.EncodeToString(buf.Bytes()),
		},
	}, nil
}

// glyphs5x7 maps each letter of "OCTOLLM" to a 5x7 bitmap, row-major, where
// '#' marks a lit pixel.
var glyphs5x7 = map[rune][]string{
	'O': {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."},
	'C': {".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."},
	'T': {"#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."},
	'L': {"#....", "#....", "#....", "#....", "#....", "#....", "#####"},
	'M': {"#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"},
}

// renderText draws text as white pixels on a black grayscale image, using a
// built-in 5x7 bitmap font scaled up to fill the canvas with a margin and
// centered. Runes without a glyph are skipped.
func renderText(text string, width, height int) *image.Gray {
	const (
		glyphW = 5
		glyphH = 7
		gap    = 1 // columns between glyphs
	)
	img := image.NewGray(image.Rect(0, 0, width, height))

	runes := []rune(text)
	cols := len(runes)*glyphW + (len(runes)-1)*gap
	if cols <= 0 {
		return img
	}

	// Largest integer scale that keeps the text within ~80% of each dimension.
	scale := int(0.8 * float64(width) / float64(cols))
	if s := int(0.8 * float64(height) / float64(glyphH)); s < scale {
		scale = s
	}
	if scale < 1 {
		scale = 1
	}

	offX := (width - cols*scale) / 2
	offY := (height - glyphH*scale) / 2

	for i, r := range runes {
		glyph, ok := glyphs5x7[r]
		if !ok {
			continue
		}
		baseX := offX + i*(glyphW+gap)*scale
		for gy := 0; gy < glyphH; gy++ {
			row := glyph[gy]
			for gx := 0; gx < glyphW; gx++ {
				if row[gx] != '#' {
					continue
				}
				for sy := 0; sy < scale; sy++ {
					for sx := 0; sx < scale; sx++ {
						img.SetGray(baseX+gx*scale+sx, offY+gy*scale+sy, color.Gray{Y: 255})
					}
				}
			}
		}
	}
	return img
}
