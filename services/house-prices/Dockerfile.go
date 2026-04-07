FROM golang:1.22-alpine AS build

WORKDIR /src
COPY serving/go/ .
RUN go mod download
RUN CGO_ENABLED=0 go build -o /proxy .

FROM alpine:3.19
COPY --from=build /proxy /proxy
EXPOSE 8080
ENTRYPOINT ["/proxy"]
