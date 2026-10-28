{{- define "llm-gateway.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llm-gateway.labels" -}}
app.kubernetes.io/name: llm-gateway
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "llm-gateway.selectorLabels" -}}
app.kubernetes.io/name: llm-gateway
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "llm-gateway.componentLabels" -}}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "llm-gateway.secretName" -}}
{{- .Values.secrets.existingSecret | default (printf "%s-secrets" (include "llm-gateway.fullname" .)) -}}
{{- end -}}

{{- define "llm-gateway.redisUrl" -}}
{{- if .Values.redis.url -}}
{{ .Values.redis.url }}
{{- else -}}
redis://{{ include "llm-gateway.fullname" . }}-redis:6379/0
{{- end -}}
{{- end -}}
