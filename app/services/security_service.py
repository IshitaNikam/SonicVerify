def get_security_level(risk_level):
    if risk_level == "LOW":
        return "LOW"

    elif risk_level == "UNCERTAIN":
        return "MEDIUM"

    elif risk_level == "HIGH":
        return "HIGH"

    else:
        return "UNKNOWN"


def get_security_action(security_level):
    if security_level == "LOW":
        return "ALLOW"

    elif security_level == "MEDIUM":
        return "WARNING"

    elif security_level == "HIGH":
        return "VERIFY"

    else:
        return "BLOCK"


def needs_verification(security_level):
    if security_level == "HIGH":
        return True

    return False


def security_check(risk_level):
    security_level = get_security_level(risk_level)

    action = get_security_action(security_level)

    verification_required = needs_verification(security_level)

    return {
        "security_level": security_level,
        "action": action,
        "verification_required": verification_required
    }
