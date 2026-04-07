// High-throughput inference proxy that sits in front of the Python serving API.
// Adds request validation, connection pooling, structured logging, and metrics.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync/atomic"
	"time"
)

type PredictionRequest struct {
	Features []float64 `json:"features"`
}

type BatchRequest struct {
	Instances [][]float64 `json:"instances"`
}

type metrics struct {
	totalRequests  atomic.Int64
	totalErrors    atomic.Int64
	totalLatencyUs atomic.Int64
}

type server struct {
	backendURL string
	client     *http.Client
	metrics    *metrics
	logger     *slog.Logger
}

func newServer(backendURL string, logger *slog.Logger) *server {
	return &server{
		backendURL: backendURL,
		client: &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		metrics: &metrics{},
		logger:  logger,
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp, err := s.client.Get(s.backendURL + "/health")
	if err != nil {
		http.Error(w, `{"status":"unhealthy","error":"backend unreachable"}`, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func (s *server) handlePredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	s.metrics.totalRequests.Add(1)

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1MB limit
	if err != nil {
		s.metrics.totalErrors.Add(1)
		http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
		return
	}

	var req PredictionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.metrics.totalErrors.Add(1)
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if len(req.Features) != 8 {
		s.metrics.totalErrors.Add(1)
		msg := fmt.Sprintf(`{"error":"expected 8 features, got %d"}`, len(req.Features))
		http.Error(w, msg, http.StatusUnprocessableEntity)
		return
	}

	s.proxy(w, "/predict", body, start)
}

func (s *server) handleBatchPredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	s.metrics.totalRequests.Add(1)

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20)) // 10MB limit
	if err != nil {
		s.metrics.totalErrors.Add(1)
		http.Error(w, `{"error":"failed to read body"}`, http.StatusBadRequest)
		return
	}

	var req BatchRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.metrics.totalErrors.Add(1)
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if len(req.Instances) == 0 {
		s.metrics.totalErrors.Add(1)
		http.Error(w, `{"error":"empty batch"}`, http.StatusBadRequest)
		return
	}

	s.proxy(w, "/predict/batch", body, start)
}

func (s *server) proxy(w http.ResponseWriter, path string, body []byte, start time.Time) {
	resp, err := s.client.Post(
		s.backendURL+path,
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		s.metrics.totalErrors.Add(1)
		s.logger.Error("backend request failed", "path", path, "error", err)
		http.Error(w, `{"error":"backend unavailable"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	latency := time.Since(start)
	s.metrics.totalLatencyUs.Add(latency.Microseconds())
	s.logger.Info("request",
		"path", path,
		"status", resp.StatusCode,
		"latency_ms", latency.Milliseconds(),
	)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	total := s.metrics.totalRequests.Load()
	errors := s.metrics.totalErrors.Load()
	latencyUs := s.metrics.totalLatencyUs.Load()

	var avgLatencyMs float64
	if total > 0 {
		avgLatencyMs = float64(latencyUs) / float64(total) / 1000.0
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"total_requests":     total,
		"total_errors":       errors,
		"avg_latency_ms":     avgLatencyMs,
	})
}

func main() {
	port := flag.String("port", "8080", "proxy listen port")
	backend := flag.String("backend", "http://localhost:8000", "Python serving backend URL")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	srv := newServer(*backend, logger)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/predict", srv.handlePredict)
	mux.HandleFunc("/predict/batch", srv.handleBatchPredict)
	mux.HandleFunc("/metrics", srv.handleMetrics)

	addr := ":" + *port
	logger.Info("starting proxy", "addr", addr, "backend", *backend)
	if err := http.ListenAndServe(addr, mux); err != nil {
		logger.Error("server failed", "error", err)
		os.Exit(1)
	}
}
